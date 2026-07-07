#!/usr/bin/env python3
"""Quantify NMT repetition degeneration in translated TTP-Eval and its impact.

Reviewer jxiH observed repetition-loop degeneration (e.g. `TTP24TTP24...`) in an
NLLB output in this repository and asked whether the multilingual experiments
(Section 7) are affected. This script answers that quantitatively:

  1. flags each translated document as degenerate or clean, where degenerate
     means it contains a short unit repeated >=10x consecutively (the NLLB
     repetition-loop failure mode) or compresses below a zlib ratio of 0.10
     (extreme redundancy) — with the English originals as the false-positive
     baseline (7/393 flagged, all genuine source-text repetition);
  2. recomputes each classifier's per-language F1 on (a) all rows, (b) the
     clean subset, (c) the degenerate subset, using the released per-sample
     predictions — no model re-runs needed;
  3. pairs every clean-subset F1 with the SAME classifier's English F1 on the
     SAME row subset, so composition effects cannot masquerade as robustness.

The question it settles: does HarmFormer's multilingual collapse survive when
translation-degenerate rows are excluded? (Answer at time of writing: yes —
paired clean-subset drops of 0.27-0.52 F1 across the six languages.)

    python scripts/analyze_translation_degeneration.py
    python scripts/analyze_translation_degeneration.py --translator nllb-200-distilled-600M
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
import sys
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
csv.field_size_limit(10_000_000)

# A short unit repeated many times back-to-back: the NLLB loop signature.
LOOP = re.compile(r"(.{2,15}?)\1{9,}")
ZLIB_FLOOR = 0.10


def is_degenerate(text: str) -> bool:
    if not text:
        return True
    if LOOP.search(text):
        return True
    return len(zlib.compress(text.encode())) / max(len(text.encode()), 1) < ZLIB_FLOOR


def f1_on(preds, golds, idx):
    tp = sum(1 for i in idx if preds[i] and golds[i])
    fp = sum(1 for i in idx if preds[i] and not golds[i])
    fn = sum(1 for i in idx if not preds[i] and golds[i])
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return 2 * p * r / (p + r) if p + r else 0.0


def per_sample(result_json: Path, setup: str):
    payload = json.loads(result_json.read_text())
    r = next(x for x in payload["results"] if x["setup"] == setup)
    ps = r["per_sample_toxic"]
    return ps["pred"], ps["gold"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--translator", default="nllb-200-3.3B",
                    choices=["nllb-200-3.3B", "nllb-200-distilled-600M"])
    ap.add_argument("--english-preds",
                    default="results/ttp_eval_baselines_with_preds/harmformer_llama_guard_en.json")
    args = ap.parse_args()

    results_dir = ROOT / ("results/ttp_eval_multilingual" if args.translator == "nllb-200-3.3B"
                          else "results/ttp_eval_multilingual_nllb600m")
    data_dir = ROOT / "data" / "TTP-Eval" / "translated" / args.translator

    with (ROOT / "data" / "TTP-Eval" / "TTPEval.tsv").open(encoding="utf-8") as f:
        eng_rows = list(csv.DictReader(f, delimiter="\t"))
    eng_degen = [is_degenerate(r.get("Body", "")) for r in eng_rows]
    print(f"Translator: {args.translator}")
    print(f"English originals flagged by the same detector: {sum(eng_degen)}/{len(eng_rows)} "
          "(false-positive baseline)\n")

    setups = {"harmformer": "HarmFormer", "llama_guard": "Llama Guard"}
    en_ps = {key: per_sample(ROOT / args.english_preds, setup) for key, setup in setups.items()}

    for key, setup in setups.items():
        files = sorted(glob.glob(str(results_dir / f"{key}_*.json")))
        if not files:
            print(f"({setup}: no result files in {results_dir})")
            continue
        print(f"=== {setup} ===")
        print(f"{'lang':<10}{'degen':>7}{'clean':>7} | {'F1 all':>7}{'F1 clean':>9}{'F1 degen':>9}"
              f" | {'EN@clean':>9}{'paired drop':>12}")
        for path in files:
            lang = os.path.basename(path).replace(f"{key}_", "").replace(".json", "")
            pred, gold = per_sample(Path(path), setup)
            en_pred, en_gold = en_ps[key]
            if gold != en_gold:
                sys.exit(f"gold ordering mismatch for {lang} — cannot pair subsets")
            tsv = data_dir / f"TTPEval_{lang}.tsv"
            with tsv.open(encoding="utf-8") as f:
                rows = list(csv.DictReader(f, delimiter="\t"))
            degen = [is_degenerate(r.get("Body", "")) for r in rows]
            n = len(rows)
            clean = [i for i in range(n) if not degen[i]]
            dirty = [i for i in range(n) if degen[i]]
            f_all = f1_on(pred, gold, range(n))
            f_cl = f1_on(pred, gold, clean)
            f_dg = f1_on(pred, gold, dirty) if dirty else float("nan")
            f_en = f1_on(en_pred, en_gold, clean)
            print(f"{lang:<10}{len(dirty):>7}{len(clean):>7} | {f_all:>7.3f}{f_cl:>9.3f}{f_dg:>9.3f}"
                  f" | {f_en:>9.3f}{f_cl - f_en:>12.3f}")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
