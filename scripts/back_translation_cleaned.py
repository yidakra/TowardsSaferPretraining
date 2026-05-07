#!/usr/bin/env python3
"""Recompute BLEU/chrF for back-translation against a *cleaned* English source.

The TTP-Eval corpus contains heavy HTML/JS/navigation clutter that NLLB
strips during forward translation, making naive corpus-BLEU between the
BT output and the noisy original implausibly low (~1 BLEU). To produce a
defensible noise-floor estimate we strip non-prose markup from the
English source and recompute corpus BLEU and chrF against that cleaned
reference. The ranking across languages is what feeds the Spearman test;
absolute values become interpretable as "round-trip retention on prose."

Output: results/back_translation/back_translation_metrics_cleaned.json
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import List


_CSS_OR_JS_HINT = re.compile(r"[{};]|\bvar\s|\bfunction\s|=>|\)\s*\{")
_HTML_TAG = re.compile(r"<[^>]+>")
_URL = re.compile(r"https?://\S+")
_MULTISPACE = re.compile(r"\s+")
_PUNCT_RUN = re.compile(r"[^\w\s,.!?'\"-]{2,}")


def _clean_text(text: str) -> str:
    """Heuristic: drop sentences that look like CSS/JS/navigation; keep prose."""
    text = _HTML_TAG.sub(" ", text)
    text = _URL.sub(" ", text)
    sentences = re.split(r"(?<=[.!?])\s+|(?<=\.)\s*\n", text)
    keep: List[str] = []
    for s in sentences:
        s = s.strip()
        if not s:
            continue
        words = s.split()
        if len(words) < 5:
            continue
        if _CSS_OR_JS_HINT.search(s):
            continue
        if _PUNCT_RUN.search(s):
            continue
        non_alpha = sum(1 for c in s if not (c.isalpha() or c.isspace()))
        if non_alpha / max(len(s), 1) > 0.25:
            continue
        keep.append(s)
    cleaned = " ".join(keep)
    cleaned = _MULTISPACE.sub(" ", cleaned).strip()
    return cleaned


def _read_tsv(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--src", default="data/TTP-Eval/TTPEval.tsv")
    p.add_argument("--bt-dir", default="results/back_translation")
    p.add_argument("--out", default="results/back_translation/back_translation_metrics_cleaned.json")
    p.add_argument("--min-cleaned-chars", type=int, default=80)
    args = p.parse_args()

    import sacrebleu

    src_rows = _read_tsv(Path(args.src))
    src_by_url = {r["URL"]: r["Body"] for r in src_rows}

    metrics: dict = {}
    for short in ("es", "fr", "de", "ar", "hi", "zh"):
        bt_path = Path(args.bt_dir) / f"back_translated_{short}.tsv"
        if not bt_path.exists():
            continue
        bt_rows = _read_tsv(bt_path)

        refs: List[str] = []
        hyps: List[str] = []
        n_dropped = 0
        for r in bt_rows:
            url = r.get("URL")
            bt = (r.get("BT_Body") or "").strip()
            if not url or url not in src_by_url:
                continue
            cleaned_ref = _clean_text(src_by_url[url])
            if len(cleaned_ref) < args.min_cleaned_chars:
                n_dropped += 1
                continue
            refs.append(cleaned_ref)
            hyps.append(bt)

        if not refs:
            continue

        bleu_clean = sacrebleu.corpus_bleu(hyps, [refs])
        chrf_clean = sacrebleu.corpus_chrf(hyps, [refs])

        # Also recompute against raw source on the same alignment for a paired
        # comparison (so the only thing that differs is the reference cleaning).
        raw_refs = [src_by_url[r["URL"]] for r in bt_rows
                    if (r.get("BT_Body") or "").strip()
                    and r.get("URL") in src_by_url
                    and len(_clean_text(src_by_url[r["URL"]])) >= args.min_cleaned_chars]
        bleu_raw = sacrebleu.corpus_bleu(hyps, [raw_refs])
        chrf_raw = sacrebleu.corpus_chrf(hyps, [raw_refs])

        metrics[short] = {
            "n_aligned_after_cleaning": len(refs),
            "n_dropped_short_after_clean": n_dropped,
            "bleu_cleaned_ref": round(bleu_clean.score, 2),
            "chrf_cleaned_ref": round(chrf_clean.score, 2),
            "bleu_raw_ref_paired": round(bleu_raw.score, 2),
            "chrf_raw_ref_paired": round(chrf_raw.score, 2),
        }
        print(f"[{short}] n={len(refs):3d}  cleaned-ref BLEU={bleu_clean.score:6.2f}  chrF={chrf_clean.score:6.2f}   "
              f"|  raw-ref BLEU={bleu_raw.score:5.2f}  chrF={chrf_raw.score:5.2f}", flush=True)

    Path(args.out).write_text(json.dumps({"per_language": metrics, "min_cleaned_chars": args.min_cleaned_chars}, indent=2))
    print(f"Saved {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
