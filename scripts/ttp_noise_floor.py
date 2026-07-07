#!/usr/bin/env python3
"""Measure the same-snapshot F1 noise floor for a TTP endpoint.

The drift experiment (Section 5, `sec:model-drift`) compares an April run against a
May A/B re-run and attributes a ~17 F1-point swing to silent `gpt-4o` snapshot drift.
That argument has one open flank: `temperature=0` is *not* bit-deterministic on
OpenAI-style endpoints (MoE routing + batching), so a reviewer can ask whether the
swing is just decoding noise. This script closes that flank by running the *same*
endpoint over TTP-Eval N times back-to-back and reporting:

  - per-run overall precision / recall / F1,
  - the F1 spread across runs (min, max, range, std) = the noise floor,
  - pairwise per-sample toxic-label disagreement counts between runs,
  - the union of samples whose toxic label ever flips across runs,
  - the distinct `system_fingerprint` values seen (via the client stats added
    alongside this script) — if the noise floor is small AND the fingerprint is
    constant, the residual April->May swing cannot be within-snapshot noise.

No GPU. Cost is ~N x the single-run TTP-Eval spend (~$5-8 each on gpt-4o).

Example (same-endpoint x3, seeded, via OpenRouter):

    python scripts/ttp_noise_floor.py \
      --setup openrouter_ttp \
      --openrouter-model openai/gpt-4o \
      --repeats 3 --seed 12345 \
      --output results/ttp_eval_noise_floor/openrouter_gpt4o_x3.json
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_loaders import TTPEvalLoader
from src.benchmarks.metrics import calculate_metrics
from src.utils.codecarbon import maybe_track_emissions
from src.utils.repro_metadata import gather_run_metadata
from src.utils.wandb import add_wandb_args, init_wandb_from_args


def _build_client(args: argparse.Namespace):
    """Instantiate the single TTP endpoint under test."""
    if args.setup == "openai_ttp":
        from src.clients.ttp_openai import OpenAITTPClient

        key = args.openai_key or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise SystemExit("openai_ttp selected but no OPENAI_API_KEY/--openai-key provided.")
        name = f"TTP ({args.openai_model})"
        return name, OpenAITTPClient(
            api_key=key,
            model=args.openai_model,
            prompt_path=args.prompt_path,
            fail_open=False,
            seed=args.seed,
        )

    if args.setup == "openrouter_ttp":
        from src.clients.ttp_openrouter import OpenRouterTTPClient

        key = args.openrouter_key or os.environ.get("OPENROUTER_API_KEY")
        if not key:
            raise SystemExit("openrouter_ttp selected but no OPENROUTER_API_KEY/--openrouter-key provided.")
        name = f"TTP (OpenRouter: {args.openrouter_model})"
        return name, OpenRouterTTPClient(
            api_key=key,
            model=args.openrouter_model,
            prompt_path=args.prompt_path,
            referer=args.openrouter_referer,
            title=args.openrouter_title,
            fail_open=False,
            seed=args.seed,
        )

    raise SystemExit(f"Unknown --setup: {args.setup}")


def _run_once(clf, samples) -> Dict[str, Any]:
    """One full pass over TTP-Eval; returns metrics + per-sample toxic booleans."""
    preds = []
    gts = []
    for s in samples:
        preds.append(clf.predict(s.body))
        gts.append(s.get_harm_label())
    metrics = calculate_metrics(predictions=preds, ground_truth=gts, dimension="toxic")
    return {
        "metrics": metrics,
        "pred_toxic": [bool(p.is_toxic()) for p in preds],
        "gold_toxic": [bool(g.is_toxic()) for g in gts],
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-path", default="data/TTP-Eval/TTPEval.tsv")
    p.add_argument("--limit", type=int, help="Evaluate only the first N samples (debug/cost control)")
    p.add_argument("--repeats", type=int, default=3, help="Number of identical passes (default 3)")
    p.add_argument("--seed", type=int, default=None, help="Decoding seed forwarded on every pass")
    p.add_argument("--output", required=True, help="Output JSON file")
    p.add_argument(
        "--setup",
        default="openrouter_ttp",
        choices=["openai_ttp", "openrouter_ttp"],
        help="Which single endpoint to stress (default openrouter_ttp)",
    )
    p.add_argument("--prompt-path", default="prompts/TTP/TTP.txt")
    p.add_argument("--openai-key")
    p.add_argument("--openai-model", default="gpt-4o")
    p.add_argument("--openrouter-key")
    p.add_argument("--openrouter-model", default=os.environ.get("OPENROUTER_MODEL", "openai/gpt-4o"))
    p.add_argument("--openrouter-referer", default=os.environ.get("OPENROUTER_REFERER"))
    p.add_argument("--openrouter-title", default=os.environ.get("OPENROUTER_TITLE"))
    add_wandb_args(p)
    args = p.parse_args()

    if args.repeats < 2:
        raise SystemExit("--repeats must be >= 2 to measure a noise floor.")

    loader = TTPEvalLoader(args.data_path)
    samples = loader.load()
    if args.limit:
        samples = samples[: args.limit]

    name, clf = _build_client(args)

    wandb_run = init_wandb_from_args(
        args,
        run_name="ttp_noise_floor",
        job_type="diagnostic",
        config={
            "data_path": args.data_path,
            "setup": args.setup,
            "repeats": args.repeats,
            "seed": args.seed,
            "openai_model": args.openai_model,
            "openrouter_model": args.openrouter_model,
        },
        extra_tags=["ttp-eval", "noise-floor", "drift"],
    )

    try:
        runs: List[Dict[str, Any]] = []
        with maybe_track_emissions(run_name=f"ttp_noise_floor_{args.setup}"):
            for i in range(args.repeats):
                print(f"[noise-floor] pass {i + 1}/{args.repeats} ...", file=sys.stderr)
                runs.append(_run_once(clf, samples))

        # Gold is identical across passes; keep one copy.
        gold = runs[0]["gold_toxic"]
        f1s = [r["metrics"]["overall"]["f1"] for r in runs]
        precisions = [r["metrics"]["overall"]["precision"] for r in runs]
        recalls = [r["metrics"]["overall"]["recall"] for r in runs]

        # Pairwise per-sample disagreement on the toxic label between passes.
        n = len(gold)
        pairwise = []
        ever_flipped = [False] * n
        for a, b in itertools.combinations(range(args.repeats), 2):
            pa, pb = runs[a]["pred_toxic"], runs[b]["pred_toxic"]
            diff_idx = [k for k in range(n) if pa[k] != pb[k]]
            for k in diff_idx:
                ever_flipped[k] = True
            pairwise.append(
                {
                    "run_a": a,
                    "run_b": b,
                    "n_disagree": len(diff_idx),
                    "disagree_rate": len(diff_idx) / n if n else 0.0,
                    "disagree_indices": diff_idx,
                }
            )
        unstable_indices = [k for k in range(n) if ever_flipped[k]]

        stats = clf.get_stats() if hasattr(clf, "get_stats") else {}

        f1_range = (max(f1s) - min(f1s)) if f1s else 0.0
        noise_floor = {
            "setup": name,
            "repeats": args.repeats,
            "n_samples": n,
            "f1_per_run": f1s,
            "precision_per_run": precisions,
            "recall_per_run": recalls,
            "f1_min": min(f1s) if f1s else 0.0,
            "f1_max": max(f1s) if f1s else 0.0,
            "f1_range": f1_range,
            "f1_stdev": statistics.pstdev(f1s) if len(f1s) > 1 else 0.0,
            "n_unstable_samples": len(unstable_indices),
            "unstable_sample_rate": len(unstable_indices) / n if n else 0.0,
            "unstable_indices": unstable_indices,
            "pairwise_disagreement": pairwise,
            "seed": args.seed,
            "system_fingerprints": stats.get("system_fingerprints"),
            "distinct_system_fingerprints": stats.get("distinct_system_fingerprints"),
            "client_stats": stats,
        }

        payload = {
            "run_metadata": gather_run_metadata(repo_root=str(Path(__file__).parent.parent)),
            "evaluation_config": {
                "dataset": args.data_path,
                "total_samples": n,
                "dimension": "toxic",
                "setup": args.setup,
                "repeats": args.repeats,
                "seed": args.seed,
                "prompt_path": args.prompt_path,
                "openai_model": args.openai_model,
                "openrouter_model": args.openrouter_model,
            },
            "noise_floor": noise_floor,
            "runs": [
                {"run": i, "metrics": r["metrics"], "pred_toxic": r["pred_toxic"]}
                for i, r in enumerate(runs)
            ],
            "gold_toxic": gold,
        }

        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

        # Console summary — the two numbers a reviewer wants side by side.
        print(f"\n[noise-floor] {name}  (n={n}, repeats={args.repeats}, seed={args.seed})")
        print(f"  F1 per run          : {[round(x, 4) for x in f1s]}")
        print(f"  F1 range (max-min)  : {f1_range:.4f}   <-- same-snapshot noise floor")
        print(f"  unstable samples    : {len(unstable_indices)}/{n} ({100 * len(unstable_indices) / n:.1f}%)")
        print(f"  distinct fingerprints: {noise_floor['distinct_system_fingerprints']}  {noise_floor['system_fingerprints']}")
        print(f"  saved               : {out_path}")

        wandb_run.update_summary(
            {
                "noise/f1_range": f1_range,
                "noise/f1_stdev": noise_floor["f1_stdev"],
                "noise/n_unstable_samples": len(unstable_indices),
                "noise/distinct_fingerprints": noise_floor["distinct_system_fingerprints"],
                "config/repeats": args.repeats,
                "output/path": str(out_path),
            }
        )
        wandb_run.log_json_artifact(out_path, name=f"ttp_noise_floor_{out_path.stem}")
    except Exception:
        wandb_run.finish(exit_code=1)
        raise
    else:
        wandb_run.finish(exit_code=0)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
