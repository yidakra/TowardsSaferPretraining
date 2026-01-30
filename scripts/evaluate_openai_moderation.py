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
from typing import Any, Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))


# NOTE: Heavy imports are deferred so `--help` is fast and does not require
# optional API dependencies or model downloads.
OpenAIModerationLoader: Any = None
OpenAIModerationSample: Any = None
calculate_metrics: Any = None
PerspectiveAPI: Any = None
LlamaGuard: Any = None
OpenAITTPClient: Any = None
OpenRouterTTPClient: Any = None
HarmFormer: Any = None
maybe_track_emissions: Any = None
HarmLabel: Any = None
Dimension: Any = None
gather_run_metadata: Any = None


def _lazy_imports() -> None:
    global OpenAIModerationLoader
    global OpenAIModerationSample
    global calculate_metrics
    global PerspectiveAPI, LlamaGuard, OpenAITTPClient, OpenRouterTTPClient
    global HarmFormer
    global maybe_track_emissions
    global HarmLabel, Dimension
    global gather_run_metadata

    if OpenAIModerationLoader is not None:
        return

    from src.data_loaders import OpenAIModerationLoader as _Loader, OpenAIModerationSample as _Sample
    from src.benchmarks.metrics import calculate_metrics as _calculate_metrics
    from src.clients import (  # heavy/optional deps live behind these imports
        PerspectiveAPI as _PerspectiveAPI,
        LlamaGuard as _LlamaGuard,
        OpenAITTPClient as _OpenAITTPClient,
        OpenRouterTTPClient as _OpenRouterTTPClient,
    )
    from src.models import HarmFormer as _HarmFormer
    from src.utils.codecarbon import maybe_track_emissions as _maybe_track_emissions
    from src.utils.taxonomy import HarmLabel as _HarmLabel, Dimension as _Dimension
    from src.utils.repro_metadata import gather_run_metadata as _gather_run_metadata

    OpenAIModerationLoader = _Loader
    OpenAIModerationSample = _Sample
    calculate_metrics = _calculate_metrics
    PerspectiveAPI = _PerspectiveAPI
    LlamaGuard = _LlamaGuard
    OpenAITTPClient = _OpenAITTPClient
    OpenRouterTTPClient = _OpenRouterTTPClient
    HarmFormer = _HarmFormer
    maybe_track_emissions = _maybe_track_emissions
    HarmLabel = _HarmLabel
    Dimension = _Dimension
    gather_run_metadata = _gather_run_metadata


def _truth_to_label(s) -> Any:
    _lazy_imports()
    if not s.is_toxic:
        return HarmLabel()
    return HarmLabel(
        hate_violence=Dimension.TOXIC,
        ideological=Dimension.TOXIC,
        sexual=Dimension.TOXIC,
        illegal=Dimension.TOXIC,
        self_inflicted=Dimension.TOXIC,
    )


def _toxic_label() -> Any:
    _lazy_imports()
    return _truth_to_label(OpenAIModerationSample(text="", is_toxic=True))


def _evaluate_binary(
    name: str,
    classifier,
    samples: List[Any],
    *,
    invalid_policy: str,
) -> Dict[str, Any]:
    _lazy_imports()
    preds: List[HarmLabel] = []
    gts: List[HarmLabel] = []
    failed = 0
    for s in samples:
        try:
            preds.append(classifier.predict(s.text))
            gts.append(_truth_to_label(s))
        except Exception:
            failed += 1
            if invalid_policy == "exclude":
                continue
            if invalid_policy == "non_toxic":
                preds.append(HarmLabel())
                gts.append(_truth_to_label(s))
                continue
            if invalid_policy == "toxic":
                preds.append(_toxic_label())
                gts.append(_truth_to_label(s))
                continue
            raise RuntimeError(f"Unknown invalid_policy: {invalid_policy}")

    metrics = calculate_metrics(predictions=preds, ground_truth=gts, dimension="toxic")

    client_stats = None
    if hasattr(classifier, "get_stats") and callable(getattr(classifier, "get_stats")):
        try:
            client_stats = classifier.get_stats()  # type: ignore[attr-defined]
        except Exception:
            client_stats = None
    return {
        "classifier": name,
        "total_samples": len(samples),
        "failed_samples": failed,
        "evaluated_samples": len(preds),
        "metrics": metrics,
        "invalid_policy": invalid_policy,
        "client_stats": client_stats,
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
            "ttp_openrouter",
            "harmformer",
        ],
        # Default to local-only baselines to avoid API calls unless explicitly requested.
        default=["llama_guard", "llama_guard_zero_shot", "llama_guard_few_shot", "harmformer"],
    )
    parser.add_argument("--output", default="results/moderation/table7_results.json", help="Output JSON path")
    parser.add_argument("--limit", type=int, help="Limit samples (debug)")

    parser.add_argument(
        "--invalid-policy",
        default=None,
        choices=["exclude", "non_toxic", "toxic"],
        help=(
            "How to handle invalid/unparseable outputs or API failures. "
            "If omitted, uses the setup's default behavior (OpenAI/OpenRouter typically fail-open; "
            "local baselines typically raise and are excluded)."
        ),
    )

    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu", "mps"], help="Device for local models")

    # API keys
    parser.add_argument("--openai-key", help="OpenAI API key (or set OPENAI_API_KEY)")
    parser.add_argument("--openrouter-key", help="OpenRouter API key (or set OPENROUTER_API_KEY)")
    parser.add_argument("--perspective-key", help="Perspective API key (or set PERSPECTIVE_API_KEY)")
    parser.add_argument("--openrouter-model", default=os.environ.get("OPENROUTER_MODEL", "openai/gpt-4o"))
    parser.add_argument("--openrouter-referer", default=os.environ.get("OPENROUTER_REFERER"), help="Optional HTTP-Referer header")
    parser.add_argument("--openrouter-title", default=os.environ.get("OPENROUTER_TITLE"), help="Optional X-Title header")


    # Paper doesn't specify Perspective threshold for Table 7; we default to 0.4 for consistency.
    parser.add_argument("--perspective-threshold", type=float, default=0.4)

    # Llama Guard
    parser.add_argument(
        "--llama-guard-model",
        default=None,
        help="HF model id override (default: meta-llama/Llama-Guard-3-8B)",
    )

    args = parser.parse_args()

    _lazy_imports()

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
                    mode="paper_table4",  # keeps behavior consistent with paper
                    paper_threshold=args.perspective_threshold,
                    paper_chunk_chars=500,
                ),
            )
        )

    # Llama Guard variants are processed sequentially below to avoid OOM
    # (loading 3 copies of 8B model would exceed A100 40GB memory)
    llama_guard_modes: List[Tuple[str, str]] = []
    if "llama_guard" in args.baselines:
        llama_guard_modes.append(("Llama Guard", "focused"))
    if "llama_guard_zero_shot" in args.baselines:
        llama_guard_modes.append(("Llama Guard Zero Shot", "zero_shot"))
    if "llama_guard_few_shot" in args.baselines:
        llama_guard_modes.append(("Llama Guard Few Shot", "few_shot"))

    if "ttp" in args.baselines:
        openai_key = args.openai_key or os.environ.get("OPENAI_API_KEY")
        if not openai_key:
            raise SystemExit("TTP enabled but no OpenAI key provided (set OPENAI_API_KEY or pass --openai-key)")
        fail_open = True if args.invalid_policy is None else (args.invalid_policy == "non_toxic")
        classifiers.append(("TTP", OpenAITTPClient(api_key=openai_key, model="gpt-4o", fail_open=fail_open)))

    if "ttp_openrouter" in args.baselines:
        key = args.openrouter_key or os.environ.get("OPENROUTER_API_KEY")
        if not key:
            raise SystemExit("TTP (OpenRouter) enabled but no OpenRouter key provided (set OPENROUTER_API_KEY or pass --openrouter-key)")
        fail_open = True if args.invalid_policy is None else (args.invalid_policy == "non_toxic")
        classifiers.append(
            (
                f"TTP (OpenRouter: {args.openrouter_model})",
                OpenRouterTTPClient(
                    api_key=key,
                    model=args.openrouter_model,
                    referer=args.openrouter_referer,
                    title=args.openrouter_title,
                    fail_open=fail_open,
                ),
            )
        )

    if "harmformer" in args.baselines:
        classifiers.append(("HarmFormer", HarmFormer(device=args.device)))

    results: List[Dict[str, Any]] = []
    all_baseline_names: List[str] = []

    invalid_policy = args.invalid_policy or "exclude"

    # Process Llama Guard variants sequentially to avoid OOM
    # (each 8B model uses ~16GB; loading 3 at once would exceed A100 40GB)
    for name, mode in llama_guard_modes:
        try:
            print(f"Loading {name} (mode={mode})...")
            clf = LlamaGuard(
                model_name=args.llama_guard_model or LlamaGuard.MODEL_NAME,
                device=args.device,
                prompt_mode=mode,
            )
            with maybe_track_emissions(run_name=f"moderation_{name.replace(' ', '_').lower()}"):
                results.append(_evaluate_binary(name, clf, samples, invalid_policy=invalid_policy))
            all_baseline_names.append(name)
            # Free GPU memory before loading next variant
            print(f"Cleaning up {name}...")
            clf.cleanup()
            del clf
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as e:
            print(f"Warning: Skipping {name} (failed): {e}")

    # Process other classifiers (non-Llama Guard)
    for name, clf in classifiers:
        with maybe_track_emissions(run_name=f"moderation_{name.replace(' ', '_').lower()}"):
            results.append(_evaluate_binary(name, clf, samples, invalid_policy=invalid_policy))
        all_baseline_names.append(name)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "run_metadata": gather_run_metadata(repo_root=str(Path(__file__).parent.parent)),
        "evaluation_config": {
            "dataset": str(args.data_path),
            "total_samples": len(samples),
            "baselines": all_baseline_names,
            "perspective_threshold": args.perspective_threshold,
            "device": args.device,
            "invalid_policy": args.invalid_policy,
        },
        "results": results,
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

