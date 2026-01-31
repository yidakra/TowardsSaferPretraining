"""Generate a simple HAVOC vs RTP leakage comparison plot.

Outputs:
  - havoc-rtp-compare.png

Data sources:
  - HAVOC: averages over all JSON files in results/havoc/*_results.json
    - RTP: reads one RTP results JSON (default: results/rtp/rtp_continuations_harmformer.json)

This is designed to keep the paper figure consistent with the updated HAVOC all-6
aggregation (Claim 4 reproducibility fix).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


METRICS = [
    ("neutral", "Neutral"),
    ("passive", "Passive"),
    ("provocative", "Provocative"),
    ("overall", "Aggregated"),
]


PAPER_ALL6_KEYS = ["gemma_2b", "gemma_9b", "gemma_27b", "llama_1b", "llama_3b", "mistral_7b"]
PAPER_MEDIUM3_KEYS = ["gemma_9b", "llama_3b", "mistral_7b"]


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _get_leakage_percentages(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Extract leakage_percentages from either top-level or nested evaluation."""
    if isinstance(payload.get("leakage_percentages"), dict):
        return payload["leakage_percentages"]
    evaluation = payload.get("evaluation")
    if isinstance(evaluation, dict) and isinstance(evaluation.get("leakage_percentages"), dict):
        return evaluation["leakage_percentages"]
    return {}


def _mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def load_havoc_means(results_dir: Path) -> Dict[str, float]:
    # Prefer the paper-faithful aggregation set when present.
    all6 = [results_dir / f"{k}_results.json" for k in PAPER_ALL6_KEYS]
    medium3 = [results_dir / f"{k}_results.json" for k in PAPER_MEDIUM3_KEYS]

    if all(p.exists() for p in all6):
        files = all6
    elif all(p.exists() for p in medium3):
        files = medium3
    else:
        # Fallback: any *_results.json, but ignore scratch/smoke artifacts.
        files = [p for p in sorted(results_dir.glob("*_results.json")) if not p.name.startswith("_")]

    if not files:
        raise FileNotFoundError(f"No HAVOC result files found in {results_dir}")

    buckets: Dict[str, List[float]] = {k: [] for k, _ in METRICS}

    for path in files:
        data = _read_json(path)
        leakage = _get_leakage_percentages(data)
        for key, _label in METRICS:
            if key in leakage and leakage[key] is not None:
                buckets[key].append(float(leakage[key]))

    means = {k: _mean(v) for k, v in buckets.items()}
    means["_n_models"] = float(len(files))
    return means


def load_rtp_metrics(rtp_json: Path) -> Dict[str, float]:
    data = _read_json(rtp_json)
    leakage = _get_leakage_percentages(data)
    out: Dict[str, float] = {}
    for key, _label in METRICS:
        if key not in leakage:
            raise KeyError(f"Missing '{key}' in leakage_percentages of {rtp_json}")
        out[key] = float(leakage[key])
    return out


def _format_pct(v: float) -> str:
    return f"{v:.1f}%"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--havoc-results-dir", default="results/havoc")
    parser.add_argument(
        "--rtp-results-json",
        default="results/rtp/rtp_continuations_harmformer_full.json",
        help="Which RTP run to compare against (default: locally generated HarmFormer-on-RTP JSON)",
    )
    parser.add_argument("--out", default="havoc-rtp-compare.png")
    args = parser.parse_args()

    rtp_path = Path(args.rtp_results_json)
    if not rtp_path.exists() and args.rtp_results_json == "results/rtp/rtp_continuations_harmformer_full.json":
        # Best-effort fallback for repos that ship a precomputed RTP JSON under a different name.
        candidates = [
            Path("results/rtp/google_gemma-2-9b_results.json"),
            Path("results/rtp/meta-llama_Llama-3_2-3B_results.json"),
            Path("results/rtp/mistralai_Mistral-7B-v0_3_results.json"),
        ]
        for c in candidates:
            if c.exists():
                rtp_path = c
                break

    havoc_means = load_havoc_means(Path(args.havoc_results_dir))
    rtp_vals = load_rtp_metrics(rtp_path)

    labels = [lab for _k, lab in METRICS]
    havoc = [havoc_means[k] for k, _lab in METRICS]
    rtp = [rtp_vals[k] for k, _lab in METRICS]

    x = list(range(len(labels)))
    width = 0.38

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.bar(
        [i - width / 2 for i in x],
        havoc,
        width=width,
        label="HAVOC",
        color="#4C78A8",
    )
    ax.bar(
        [i + width / 2 for i in x],
        rtp,
        width=width,
        label="RTP",
        color="#F58518",
        hatch="///",
        edgecolor="#F58518",
        linewidth=0.8,
    )

    ax.set_xticks(x, labels)
    ax.set_ylabel("Leakage (%)")
    ax.set_ylim(0, max(max(havoc), max(rtp)) * 1.25)
    ax.set_title("Leakage rates: HAVOC vs RealToxicityPrompts")

    # Average lines (paper-style) using the Aggregated/overall bucket.
    havoc_avg = float(havoc_means.get("overall", float("nan")))
    rtp_avg = float(rtp_vals.get("overall", float("nan")))
    if havoc_avg == havoc_avg:  # not NaN
        ax.axhline(havoc_avg, color="#4C78A8", linestyle="--", linewidth=1.2, label=f"HAVOC avg = {_format_pct(havoc_avg)}")
    if rtp_avg == rtp_avg:  # not NaN
        ax.axhline(rtp_avg, color="#F58518", linestyle="--", linewidth=1.2, label=f"RTP avg = {_format_pct(rtp_avg)}")

    ax.legend(frameon=False)

    # Annotate
    for i, (h, r) in enumerate(zip(havoc, rtp, strict=True)):
        ax.text(i - width / 2, h + 0.8, f"{h:.2f}", ha="center", va="bottom", fontsize=9)
        ax.text(i + width / 2, r + 0.8, f"{r:.2f}", ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    fig.savefig(Path(args.out), dpi=200)


if __name__ == "__main__":
    main()
