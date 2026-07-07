#!/usr/bin/env python3
"""Build a human-annotation sheet for the multilingual label-transfer spot-check.

Reviewer strengthening request: validate that the English Safe/Topical/Toxic
labels still hold after NLLB translation, on a small human-annotated sample in
two languages. This does NOT do the judging (that is the human's job) — it just
assembles a stratified, reproducible sheet that pairs each translated document
with its transferred English label and leaves blank columns for the annotator.

For each selected TTP-Eval row it emits: the row index, the document-level gold
toxic flag, the five per-category English labels (None/Topical/Intent), the
English body and its translation (both flattened + truncated for spreadsheet
use), and empty `label_still_holds` / `which_labels_changed` / `notes` columns.

Sampling is stratified (half originally-toxic, half not) and deterministic
(fixed seed), so the same sheet regenerates byte-for-byte.

Example:
    python scripts/make_label_transfer_sheet.py \
      --langs spa_Latn zho_Hans --per-lang 20 \
      --output results/label_transfer_audit/label_transfer_sheet.tsv
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path
from typing import List

ROOT = Path(__file__).resolve().parents[1]
ENGLISH_TSV = ROOT / "data" / "TTP-Eval" / "TTPEval.tsv"
TRANSLATED_DIR = ROOT / "data" / "TTP-Eval" / "translated" / "nllb-200-3.3B"

LABEL_COLS = ["Hate&V Lab", "Ideologi Lab", "Sexual Lab", "Illegal Lab", "Self-Infl Lab"]
LABEL_SHORT = ["H", "IH", "SE", "IL", "SI"]


def _read_tsv(path: Path) -> List[dict]:
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def _is_toxic(row: dict) -> bool:
    """Document is toxic if any harm category carries the Intent label.

    Uses the project's own parser so alternate encodings (e.g. S2) match the
    evaluation loader exactly.
    """
    sys.path.insert(0, str(ROOT))
    from src.utils.taxonomy import Dimension
    return any(Dimension.from_label(row.get(c) or "") == Dimension.TOXIC for c in LABEL_COLS)


def _flatten(text: str, limit: int) -> str:
    """Collapse whitespace so the cell stays on one spreadsheet line, then truncate."""
    flat = " ".join((text or "").split())
    return flat[:limit] + (" […]" if len(flat) > limit else "")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--langs", nargs="+", default=["spa_Latn", "zho_Hans"],
                   help="NLLB language codes matching TTPEval_<code>.tsv (default: spa_Latn zho_Hans)")
    p.add_argument("--per-lang", type=int, default=20, help="Samples per language (half toxic, half not)")
    p.add_argument("--body-chars", type=int, default=900, help="Max chars of each body cell")
    p.add_argument("--seed", type=int, default=20260703)
    p.add_argument("--output", default="results/label_transfer_audit/label_transfer_sheet.tsv")
    args = p.parse_args()

    english = _read_tsv(ENGLISH_TSV)
    # Stable original row index (1-based, matching per-sample TSV / loader order).
    for i, row in enumerate(english):
        row["_idx"] = i

    rng = random.Random(args.seed)
    half = args.per_lang // 2

    out_rows: List[dict] = []
    for lang in args.langs:
        tpath = TRANSLATED_DIR / f"TTPEval_{lang}.tsv"
        if not tpath.exists():
            raise SystemExit(f"Missing translation file: {tpath}")
        translated = _read_tsv(tpath)
        if len(translated) != len(english):
            raise SystemExit(f"Row-count mismatch for {lang}: {len(translated)} vs {len(english)} English")

        toxic_idx = [r["_idx"] for r in english if _is_toxic(r)]
        safe_idx = [r["_idx"] for r in english if not _is_toxic(r)]
        rng.shuffle(toxic_idx)
        rng.shuffle(safe_idx)
        picked = sorted(toxic_idx[:half] + safe_idx[: args.per_lang - half])

        for idx in picked:
            en, tr = english[idx], translated[idx]
            labels = {short: (en.get(col) or "").strip() for short, col in zip(LABEL_SHORT, LABEL_COLS)}
            out_rows.append({
                "lang": lang,
                "row_index": idx,
                "doc_toxic_gold": "toxic" if _is_toxic(en) else "non-toxic",
                **{f"{s}_lab": labels[s] for s in LABEL_SHORT},
                "english_body": _flatten(en.get("Body", ""), args.body_chars),
                "translated_body": _flatten(tr.get("Body", ""), args.body_chars),
                # --- annotator fills these in ---
                "label_still_holds": "",       # Y / N / Partial
                "which_labels_changed": "",    # e.g. "SI: Intent->Topical"
                "notes": "",
            })

    out_path = ROOT / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "lang", "row_index", "doc_toxic_gold",
        *[f"{s}_lab" for s in LABEL_SHORT],
        "english_body", "translated_body",
        "label_still_holds", "which_labels_changed", "notes",
    ]
    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t", quoting=csv.QUOTE_ALL)
        w.writeheader()
        w.writerows(out_rows)

    n_tox = sum(1 for r in out_rows if r["doc_toxic_gold"] == "toxic")
    print(f"wrote {out_path}")
    print(f"  languages : {', '.join(args.langs)}")
    print(f"  rows      : {len(out_rows)} ({n_tox} toxic / {len(out_rows) - n_tox} non-toxic)")
    print("  annotator columns to fill: label_still_holds, which_labels_changed, notes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
