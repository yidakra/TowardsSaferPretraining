#!/usr/bin/env python3
"""Run multilingual extension evaluation on translated TTP-Eval.

This script is a thin, reproducible wrapper that:
  1) iterates over translated TTP-Eval TSVs under data/TTP-Eval/translated/<translator>/
  2) runs `scripts/evaluate_ttp_eval.py` for HarmFormer and/or Llama Guard
  3) writes one JSON per (setup, language) to a stable location consumed by:
     - scripts/plot_multilingual_f1.py

Outputs (by default):
  - results/ttp_eval_multilingual/harmformer_<lang>.json
  - results/ttp_eval_multilingual/llama_guard_<lang>.json

This repo does not version `results/`.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List


DEFAULT_LANGS = ["spa_Latn", "fra_Latn", "deu_Latn", "arb_Arab", "hin_Deva", "zho_Hans"]


def _run(cmd: List[str]) -> None:
    print("+ " + " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> int:
    p = argparse.ArgumentParser(description="Evaluate HarmFormer/Llama Guard on translated TTP-Eval TSVs")
    p.add_argument(
        "--translated-dir",
        default="data/TTP-Eval/translated/nllb-200-3.3B",
        help="Directory containing TTPEval_<lang>.tsv translated files",
    )
    p.add_argument(
        "--langs",
        nargs="+",
        default=DEFAULT_LANGS,
        help="NLLB language codes to evaluate (matches TTPEval_<lang>.tsv filenames)",
    )
    p.add_argument(
        "--setups",
        nargs="+",
        default=["harmformer", "llama_guard"],
        choices=["harmformer", "llama_guard"],
        help="Which local setups to run",
    )
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu", "mps"], help="Device for local models")
    p.add_argument("--limit", type=int, help="Optional sample limit per language")
    p.add_argument(
        "--output-dir",
        default="results/ttp_eval_multilingual",
        help="Where to write per-language JSON results",
    )
    p.add_argument(
        "--dimension",
        default="toxic",
        choices=["toxic", "topical", "all"],
        help="Which dimension to evaluate",
    )
    args = p.parse_args()

    translated_dir = Path(args.translated_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for lang in args.langs:
        tsv_path = translated_dir / f"TTPEval_{lang}.tsv"
        if not tsv_path.exists():
            raise FileNotFoundError(f"Missing translated TSV: {tsv_path}")

        common = [
            sys.executable,
            "scripts/evaluate_ttp_eval.py",
            "--data-path",
            str(tsv_path),
            "--device",
            args.device,
            "--dimension",
            args.dimension,
            "--invalid-policy",
            "exclude",
        ]
        if args.limit is not None:
            common += ["--limit", str(args.limit)]

        if "harmformer" in args.setups:
            out_path = out_dir / f"harmformer_{lang}.json"
            _run(common + ["--setups", "harmformer", "--output", str(out_path)])

        if "llama_guard" in args.setups:
            out_path = out_dir / f"llama_guard_{lang}.json"
            _run(common + ["--setups", "llama_guard", "--output", str(out_path)])

    print(f"Done. Wrote results to: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
