"""Generate the April->May TTP drift timeline figure for both GPT-4o-judged
benchmarks (TTP-Eval n=393 and OpenAI Moderation n=1680).

Inputs (no API spend):
  - results/ttp_eval_endpoint_ab/ttp_openai_vs_openrouter.json (per-sample preds for May)
  - April point estimates from the paper text (P, R, F1 reported in tex Section 5)
  - results/moderation/table7_may_rerun_openrouter_v2.json (May OpenAI-Mod metrics)

Bootstrap CIs: 10,000 resamples on the per-sample contingency vector. For
April runs and the OpenAI-Mod May run, we reconstruct an exact contingency
table from the published P, R, F1 and gold positive count, then bootstrap
from the implied per-sample outcome vector (TP/FP/FN/TN cells). For the
TTP-Eval May AB runs we use the actual per-sample preds.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT = ROOT / "report" / "drift-timeline.png"


def bootstrap_f1_ci(outcomes: np.ndarray, n_boot: int = 10_000, seed: int = 0) -> tuple[float, float, float]:
    """outcomes: int array of length n with values 0=TN, 1=FP, 2=FN, 3=TP.
    Returns (f1_point, ci_lo, ci_hi).
    """
    rng = np.random.default_rng(seed)
    n = len(outcomes)
    idx = rng.integers(0, n, size=(n_boot, n))
    f1s = np.empty(n_boot)
    for i in range(n_boot):
        s = outcomes[idx[i]]
        tp = (s == 3).sum()
        fp = (s == 1).sum()
        fn = (s == 2).sum()
        denom = 2 * tp + fp + fn
        f1s[i] = 0.0 if denom == 0 else (2 * tp) / denom
    tp = (outcomes == 3).sum()
    fp = (outcomes == 1).sum()
    fn = (outcomes == 2).sum()
    denom = 2 * tp + fp + fn
    f1_point = 0.0 if denom == 0 else (2 * tp) / denom
    return float(f1_point), float(np.percentile(f1s, 2.5)), float(np.percentile(f1s, 97.5))


def outcomes_from_counts(tp: int, fp: int, fn: int, tn: int) -> np.ndarray:
    return np.concatenate([np.full(tn, 0), np.full(fp, 1), np.full(fn, 2), np.full(tp, 3)]).astype(int)


def outcomes_from_pred_gold(pred: list[bool], gold: list[bool]) -> np.ndarray:
    out = np.zeros(len(pred), dtype=int)
    for i, (p, g) in enumerate(zip(pred, gold)):
        if p and g:
            out[i] = 3
        elif p and not g:
            out[i] = 1
        elif (not p) and g:
            out[i] = 2
        else:
            out[i] = 0
    return out


def main():
    # -----------------------------------------------------------------
    # TTP-Eval (n=393, G+ = 95, G- = 298)
    # -----------------------------------------------------------------
    # April: P=0.92, R=0.46, F1=0.62 -> TP=44, FP=4, FN=51, TN=294
    ttp_april = bootstrap_f1_ci(outcomes_from_counts(tp=44, fp=4, fn=51, tn=294), seed=1)

    # May: load real per-sample preds
    ab = json.loads((RESULTS / "ttp_eval_endpoint_ab" / "ttp_openai_vs_openrouter.json").read_text())
    ttp_may_or = None  # OpenRouter
    ttp_may_oai = None  # OpenAI direct
    for r in ab["results"]:
        outc = outcomes_from_pred_gold(r["per_sample_toxic"]["pred"], r["per_sample_toxic"]["gold"])
        ci = bootstrap_f1_ci(outc, seed=2)
        if "OpenRouter" in r["setup"]:
            ttp_may_or = ci
        else:
            ttp_may_oai = ci

    # -----------------------------------------------------------------
    # OpenAI Moderation (n=1680, G+ = 522, G- = 1158)
    # -----------------------------------------------------------------
    # April: P=0.80, R=0.42, F1=0.55 -> TP=219, FP=55, FN=303, TN=1103
    mod_april = bootstrap_f1_ci(outcomes_from_counts(tp=219, fp=55, fn=303, tn=1103), seed=3)
    # May (v2 json): P=0.7515, R=0.9329, F1=0.8325; n=1680 evaluated, 0 failed
    # TP = round(0.9329 * 522) = 487; FP = round(487/0.7515 - 487) = 161
    # FN = 522 - 487 = 35; TN = 1158 - 161 = 997
    mod_may = bootstrap_f1_ci(outcomes_from_counts(tp=487, fp=161, fn=35, tn=997), seed=4)

    # -----------------------------------------------------------------
    # Plot: 2 panels, one per benchmark
    # -----------------------------------------------------------------
    plt.rcParams.update({
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "legend.fontsize": 8,
        "font.family": "serif",
    })

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0), sharey=True)

    # Panel 1: TTP-Eval
    ax = axes[0]
    paper = 0.83
    ax.axhline(paper, ls="--", color="black", lw=1.0, alpha=0.65, zorder=1)
    ax.text(0.02, paper + 0.012, f"paper (0.83)", fontsize=7.5, transform=ax.get_yaxis_transform(), color="black")

    points = [
        ("April\nOpenRouter", ttp_april, "#c0504d"),
        ("May\nOpenRouter", ttp_may_or, "#4f81bd"),
        ("May\nOpenAI direct", ttp_may_oai, "#4f81bd"),
    ]
    xs = np.arange(len(points))
    for i, (label, (f1, lo, hi), color) in enumerate(points):
        ax.errorbar(i, f1, yerr=[[f1 - lo], [hi - f1]], fmt="o", color=color,
                    capsize=4, lw=1.4, markersize=6, zorder=3)
        ax.text(i, hi + 0.018, f"{f1:.2f}", ha="center", fontsize=7.5)
    ax.set_xticks(xs)
    ax.set_xticklabels([p[0] for p in points])
    ax.set_ylabel("Document-level F1")
    ax.set_title("TTP-Eval ($n=393$)")
    ax.set_ylim(0.40, 0.95)
    ax.grid(axis="y", ls=":", alpha=0.4)

    # Panel 2: OpenAI Moderation
    ax = axes[1]
    paper = 0.80
    ax.axhline(paper, ls="--", color="black", lw=1.0, alpha=0.65, zorder=1)
    ax.text(0.02, paper + 0.012, f"paper (0.80)", fontsize=7.5, transform=ax.get_yaxis_transform(), color="black")

    points = [
        ("April\nOpenRouter", mod_april, "#c0504d"),
        ("May\nOpenRouter", mod_may, "#4f81bd"),
    ]
    xs = np.arange(len(points))
    for i, (label, (f1, lo, hi), color) in enumerate(points):
        ax.errorbar(i, f1, yerr=[[f1 - lo], [hi - f1]], fmt="o", color=color,
                    capsize=4, lw=1.4, markersize=6, zorder=3)
        ax.text(i, hi + 0.012, f"{f1:.2f}", ha="center", fontsize=7.5)
    ax.set_xticks(xs)
    ax.set_xticklabels([p[0] for p in points])
    ax.set_title("OpenAI Moderation ($n=1{,}680$)")
    ax.grid(axis="y", ls=":", alpha=0.4)

    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=200, bbox_inches="tight")
    print(f"wrote {OUT}")
    print(f"TTP-Eval April     : F1={ttp_april[0]:.3f} CI=[{ttp_april[1]:.3f}, {ttp_april[2]:.3f}]")
    print(f"TTP-Eval May (OR)  : F1={ttp_may_or[0]:.3f} CI=[{ttp_may_or[1]:.3f}, {ttp_may_or[2]:.3f}]")
    print(f"TTP-Eval May (OAI) : F1={ttp_may_oai[0]:.3f} CI=[{ttp_may_oai[1]:.3f}, {ttp_may_oai[2]:.3f}]")
    print(f"OpenAI Mod April   : F1={mod_april[0]:.3f} CI=[{mod_april[1]:.3f}, {mod_april[2]:.3f}]")
    print(f"OpenAI Mod May     : F1={mod_may[0]:.3f} CI=[{mod_may[1]:.3f}, {mod_may[2]:.3f}]")


if __name__ == "__main__":
    main()
