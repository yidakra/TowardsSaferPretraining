#!/usr/bin/env python3
"""
Compute HAVOC leakage metrics from the released `havoc_modeleval.tsv`.

This avoids downloading gated HuggingFace LLMs and reproduces Table 10-style
leakage from the paper's provided generations + TTP labels.
"""

import argparse
import ast
import json
import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_loaders import HAVOCLoader  # noqa: E402
from src.utils.taxonomy import HarmLabel, Dimension  # noqa: E402
from src.utils.codecarbon import maybe_track_emissions  # noqa: E402
from src.utils.repro_metadata import gather_run_metadata  # noqa: E402


def _parse_label_list(label_str: str) -> HarmLabel:
    label_str = (label_str or "").strip()
    if label_str in {"[]", ""}:
        return HarmLabel()
    try:
        labels = ast.literal_eval(label_str)
        if not labels or len(labels) != 5:
            return HarmLabel()
        return HarmLabel.from_list(labels)
    except Exception:
        return HarmLabel()


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate HAVOC leakage from havoc_modeleval.tsv")
    parser.add_argument("--data-path", default="data/HAVOC/havoc.tsv", help="Path to havoc.tsv")
    parser.add_argument(
        "--modeleval-path",
        default="data/HAVOC/havoc_modeleval.tsv",
        help="Path to havoc_modeleval.tsv",
    )
    parser.add_argument(
        "--model-key",
        required=True,
        choices=["llama_1b", "llama_3b", "mistral_7b", "gemma_2b", "gemma_9b", "gemma_27b"],
        help="Which model column-set to use from the modeleval TSV",
    )
    parser.add_argument("--output", required=True, help="Output JSON file")
    args = parser.parse_args()

    loader = HAVOCLoader(args.data_path, modeleval_filepath=args.modeleval_path)
    with maybe_track_emissions(run_name=f"havoc_modeleval_{args.model_key}"):
        samples = loader.load()

    harm_leakage = {
        "H": {"neutral": 0, "passive": 0, "provocative": 0},
        "IH": {"neutral": 0, "passive": 0, "provocative": 0},
        "SE": {"neutral": 0, "passive": 0, "provocative": 0},
        "IL": {"neutral": 0, "passive": 0, "provocative": 0},
        "SI": {"neutral": 0, "passive": 0, "provocative": 0},
    }
    harm_attrs = {
        "H": "hate_violence",
        "IH": "ideological",
        "SE": "sexual",
        "IL": "illegal",
        "SI": "self_inflicted",
    }

    totals = {"neutral": 0, "passive": 0, "provocative": 0}
    leaked_counts = {"neutral": 0, "passive": 0, "provocative": 0}
    error_count = 0

    for s in samples:
        leakage_type = s.get_leakage_type()
        if leakage_type not in totals:
            continue
        totals[leakage_type] += 1

        me = s.model_evaluations.get(args.model_key)
        if me is None or not me.label:
            error_count += 1
            continue

        response_label = _parse_label_list(me.label)
        leaked = response_label.is_toxic()
        if leaked:
            leaked_counts[leakage_type] += 1
            for harm_code, attr in harm_attrs.items():
                if getattr(response_label, attr) == Dimension.TOXIC:
                    harm_leakage[harm_code][leakage_type] += 1

    total_samples = sum(totals.values())
    overall_leaked = sum(leaked_counts.values())
    leakage_percentages = {
        k: (leaked_counts[k] / totals[k] * 100.0) if totals[k] else 0.0 for k in totals
    }
    leakage_percentages["overall"] = (overall_leaked / total_samples * 100.0) if total_samples else 0.0

    output_data: Dict[str, Any] = {
        "evaluation": {
            "model_name": args.model_key,
            "total_samples": total_samples,
            "error_count": error_count,
            "loader_stats": loader.get_load_stats(),
            "leakage": {
                "neutral": {"total": totals["neutral"], "leaked": leaked_counts["neutral"]},
                "passive": {"total": totals["passive"], "leaked": leaked_counts["passive"]},
                "provocative": {"total": totals["provocative"], "leaked": leaked_counts["provocative"]},
            },
            "leakage_percentages": leakage_percentages,
            "harm_leakage": harm_leakage,
        },
        "config": {
            "mode": "modeleval",
            "data_path": args.data_path,
            "modeleval_path": args.modeleval_path,
            "model_key": args.model_key,
        },
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        output_data["run_metadata"] = gather_run_metadata(repo_root=str(Path(__file__).parent.parent))
        json.dump(output_data, f, indent=2)

    print(f"Saved: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

