#!/usr/bin/env python3
"""
Evaluate safety classifiers on the OpenAI Moderation test set (samples-1680).

Paper mapping (Mendu et al., 2025 - Table 7):
The paper evaluates Perspective, Llama Guard (prompt variants), TTP, and HarmFormer
on the OpenAI Moderation test set and reports precision/recall/F1 on a single
binary toxic label (522 toxic / 1158 non-toxic).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_loaders import OpenAIModerationLoader, OpenAIModerationSample  # noqa: E402
from src.benchmarks.metrics import calculate_metrics  # noqa: E402
from src.clients import PerspectiveAPI, LlamaGuard  # noqa: E402
from src.clients.ttp_openai import OpenAITTPClient  # noqa: E402
from src.models import HarmFormer  # noqa: E402
from src.utils.codecarbon import maybe_track_emissions  # noqa: E402
from src.utils.taxonomy import HarmLabel, Dimension  # noqa: E402


def _truth_to_label(s: OpenAIModerationSample) -> HarmLabel:
    if not s.is_toxic:
        return HarmLabel()
    return HarmLabel(
        hate_violence=Dimension.TOXIC,
        ideological=Dimension.TOXIC,
        sexual=Dimension.TOXIC,
        illegal=Dimension.TOXIC,
        self_inflicted=Dimension.TOXIC,
    )


def _evaluate_binary(
    name: str,
    classifier,
    samples: List[OpenAIModerationSample],
) -> Dict[str, Any]:
    preds: List[HarmLabel] = []
    gts: List[HarmLabel] = []
    failed = 0
    for s in samples:
        try:
            preds.append(classifier.predict(s.text))
            gts.append(_truth_to_label(s))
        except Exception:
            failed += 1
            continue

    metrics = calculate_metrics(predictions=preds, ground_truth=gts, dimension="toxic")
    return {
        "classifier": name,
        "total_samples": len(samples),
        "failed_samples": failed,
        "evaluated_samples": len(preds),
        "metrics": metrics,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate on OpenAI Moderation test set (Table 7)")
    parser.add_argument(
        "--data-path",
        default="data/moderation-api-release/data/samples-1680.jsonl.gz",
        help="Path to samples-1680.jsonl.gz (from openai/moderation-api-release)",
    )
    parser.add_argument(
        "--baselines",
        nargs="+",
        choices=[
            "perspective",
            "llama_guard",
            "llama_guard_zero_shot",
            "llama_guard_few_shot",
            "ttp",
            "harmformer",
        ],
        default=["perspective", "llama_guard", "llama_guard_zero_shot", "llama_guard_few_shot", "ttp", "harmformer"],
    )
    parser.add_argument("--output", default="results/moderation/table7_results.json", help="Output JSON path")
    parser.add_argument("--limit", type=int, help="Limit samples (debug)")

    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu", "mps"], help="Device for local models")

    # API keys
    parser.add_argument("--openai-key", help="OpenAI API key (or set OPENAI_API_KEY)")
    parser.add_argument("--perspective-key", help="Perspective API key (or set PERSPECTIVE_API_KEY)")

    # Paper doesn't specify Perspective threshold for Table 7; we default to 0.4 for consistency.
    parser.add_argument("--perspective-threshold", type=float, default=0.4)

    # Llama Guard
    parser.add_argument("--llama-guard-model", default=None, help="HF model id override (default: meta-llama/LlamaGuard-7b)")

    args = parser.parse_args()

    loader = OpenAIModerationLoader(args.data_path)
    samples = loader.load(limit=args.limit)

    # Validate class balance when running full set
    if args.limit is None:
        toxic = sum(1 for s in samples if s.is_toxic)
        non_toxic = len(samples) - toxic
        print(f"Loaded {len(samples)} samples (toxic={toxic}, non_toxic={non_toxic})")

    classifiers: List[Tuple[str, Any]] = []

    if "perspective" in args.baselines:
        perspective_key = args.perspective_key or os.environ.get("PERSPECTIVE_API_KEY")
        if not perspective_key and os.environ.get("ENABLE_PERSPECTIVE_WITH_GEMINI_KEY") == "1":
            perspective_key = os.environ.get("GEMINI_API_KEY")
        if not perspective_key:
            raise SystemExit("Perspective enabled but no key provided (set PERSPECTIVE_API_KEY or pass --perspective-key)")
        classifiers.append(
            (
                "Perspective API",
                PerspectiveAPI(
                    api_key=perspective_key,
                    mode="paper_table4",  # chunking not harmful for short texts; keeps behavior consistent
                    paper_threshold=args.perspective_threshold,
                    paper_chunk_chars=500,
                ),
            )
        )

    if "llama_guard" in args.baselines:
        classifiers.append(
            (
                "Llama Guard",
                LlamaGuard(model_name=args.llama_guard_model or LlamaGuard.MODEL_NAME, device=args.device, prompt_mode="focused"),
            )
        )
    if "llama_guard_zero_shot" in args.baselines:
        classifiers.append(
            (
                "Llama Guard Zero Shot",
                LlamaGuard(model_name=args.llama_guard_model or LlamaGuard.MODEL_NAME, device=args.device, prompt_mode="zero_shot"),
            )
        )
    if "llama_guard_few_shot" in args.baselines:
        classifiers.append(
            (
                "Llama Guard Few Shot",
                LlamaGuard(model_name=args.llama_guard_model or LlamaGuard.MODEL_NAME, device=args.device, prompt_mode="few_shot"),
            )
        )

    if "ttp" in args.baselines:
        openai_key = args.openai_key or os.environ.get("OPENAI_API_KEY")
        if not openai_key:
            raise SystemExit("TTP enabled but no OpenAI key provided (set OPENAI_API_KEY or pass --openai-key)")
        classifiers.append(("TTP", OpenAITTPClient(api_key=openai_key, model="gpt-4o")))

    if "harmformer" in args.baselines:
        classifiers.append(("HarmFormer", HarmFormer(device=args.device)))

    results: List[Dict[str, Any]] = []
    for name, clf in classifiers:
        with maybe_track_emissions(run_name=f"moderation_{name.replace(' ', '_').lower()}"):
            results.append(_evaluate_binary(name, clf, samples))

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "evaluation_config": {
            "dataset": str(args.data_path),
            "total_samples": len(samples),
            "baselines": [n for n, _ in classifiers],
            "perspective_threshold": args.perspective_threshold,
            "device": args.device,
        },
        "results": results,
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

