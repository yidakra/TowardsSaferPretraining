#!/usr/bin/env python3
"""
Aggregate HAVOC results across all models.

This script expects outputs from `scripts/evaluate_havoc.py`, which write JSON with:
  {
    "evaluation": { "leakage_percentages": { "neutral": ..., "passive": ..., "provocative": ..., "overall": ... }, ... },
    "config": {...},
    "samples": [...]
  }
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


def _safe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(x)
    except (TypeError, ValueError):
        return None


def _extract_leakage_percentages(blob: Dict[str, Any]) -> Optional[Dict[str, float]]:
    # Preferred schema (evaluate_havoc.py)
    evaluation = blob.get("evaluation")
    if isinstance(evaluation, dict):
        lp = evaluation.get("leakage_percentages")
        if isinstance(lp, dict):
            out = {}
            for k in ["neutral", "passive", "provocative", "overall"]:
                v = _safe_float(lp.get(k))
                if v is None:
                    return None
                out[k] = v
            return out

    # Backwards-compat fallback (older ad-hoc keys)
    keys = {
        "neutral": "neutral_leak",
        "passive": "passive_leak",
        "provocative": "provocative_leak",
        "overall": "overall_leak",
    }
    out2 = {}
    for k, legacy in keys.items():
        v = _safe_float(blob.get(legacy))
        if v is None:
            return None
        out2[k] = v
    return out2


def main() -> int:
    parser = argparse.ArgumentParser(description="Aggregate HAVOC *_results.json files")
    parser.add_argument("--input-dir", default="results/havoc", help="Directory containing *_results.json")
    parser.add_argument("--output", default="results/havoc/aggregated_results.json", help="Output JSON path")
    args = parser.parse_args()

    results_dir = Path(args.input_dir)
    if not (results_dir.exists() and results_dir.is_dir()):
        raise FileNotFoundError(
            f"Results directory '{results_dir}' not found. Please ensure HAVOC evaluation has been run first."
        )

    per_model: List[Dict[str, Any]] = []
    for result_file in sorted(results_dir.glob("*_results.json")):
        try:
            with open(result_file) as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            print(f"Error processing {result_file}: {e}")
            continue

        lp = _extract_leakage_percentages(data if isinstance(data, dict) else {})
        if lp is None:
            print(f"Warning: could not extract leakage_percentages from {result_file}")
            continue

        per_model.append(
            {
                "model": result_file.stem.replace("_results", ""),
                "leakage_percentages": lp,
            }
        )

    if not per_model:
        print("No valid results found for pattern *_results.json")
        return 1

    # Aggregate averages
    totals = {"neutral": 0.0, "passive": 0.0, "provocative": 0.0, "overall": 0.0}
    for r in per_model:
        for k in totals:
            totals[k] += float(r["leakage_percentages"][k])

    n = float(len(per_model))
    averages = {k: totals[k] / n for k in totals}

    print("\n=== HAVOC Leakage Summary (percent) ===")
    for r in per_model:
        lp = r["leakage_percentages"]
        print(
            f"- {r['model']}: neutral={lp['neutral']:.2f} passive={lp['passive']:.2f} "
            f"provocative={lp['provocative']:.2f} overall={lp['overall']:.2f}"
        )
    print(
        f"\nAverage: neutral={averages['neutral']:.2f} passive={averages['passive']:.2f} "
        f"provocative={averages['provocative']:.2f} overall={averages['overall']:.2f}"
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump({"per_model": per_model, "averages": averages}, f, indent=2)
    print(f"\nAggregated results saved to: {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
