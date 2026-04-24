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
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import List, Dict, Any

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.utils.codecarbon import maybe_track_emissions
from src.utils.wandb import add_wandb_args, init_wandb_from_args, extract_overall_metrics


DEFAULT_LANGS = ["spa_Latn", "fra_Latn", "deu_Latn", "arb_Arab", "hin_Deva", "zho_Hans"]


def _run(cmd: List[str], *, disable_wandb: bool = False) -> None:
    print("+ " + " ".join(cmd))
    env = None
    if disable_wandb:
        env = os.environ.copy()
        env["WANDB_ENABLED"] = "0"
        env["WANDB_MODE"] = "disabled"
    subprocess.run(cmd, check=True, env=env)


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
    add_wandb_args(p)
    args = p.parse_args()

    translated_dir = Path(args.translated_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    collected_results: Dict[str, Dict[str, Any]] = {}

    with maybe_track_emissions(run_name="ttp_eval_multilingual"):
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
                _run(common + ["--setups", "harmformer", "--output", str(out_path)], disable_wandb=True)
                if out_path.exists():
                    payload = json.loads(out_path.read_text(encoding="utf-8"))
                    collected_results[f"harmformer_{lang}"] = extract_overall_metrics(payload)

            if "llama_guard" in args.setups:
                out_path = out_dir / f"llama_guard_{lang}.json"
                _run(common + ["--setups", "llama_guard", "--output", str(out_path)], disable_wandb=True)
                if out_path.exists():
                    payload = json.loads(out_path.read_text(encoding="utf-8"))
                    collected_results[f"llama_guard_{lang}"] = extract_overall_metrics(payload)

    wandb_run = init_wandb_from_args(
        args,
        run_name="evaluate_ttp_eval_multilingual",
        job_type="evaluation",
        config={
            "translated_dir": str(translated_dir),
            "langs": args.langs,
            "setups": args.setups,
            "device": args.device,
            "limit": args.limit,
            "dimension": args.dimension,
            "output_dir": str(out_dir),
        },
        extra_tags=["multilingual", "ttp-eval", "reproduction"],
    )
    try:
        for key, metrics in collected_results.items():
            flat = {f"{key}/{k}": v for k, v in metrics.items()}
            wandb_run.update_summary(flat)
        wandb_run.update_summary({"output/count": len(collected_results), "output/dir": str(out_dir)})
        for key in collected_results:
            path = out_dir / f"{key}.json"
            if path.exists():
                wandb_run.log_json_artifact(path, name=f"ttp_eval_multilingual_{path.stem}")
    finally:
        wandb_run.finish()

    print(f"Done. Wrote results to: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
