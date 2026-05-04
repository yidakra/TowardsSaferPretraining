#!/usr/bin/env python3
"""Bootstrap 95% confidence intervals for binary toxic precision/recall/F1.

Reads a TTP-Eval result JSON written by evaluate_ttp_eval.py and emits a
nested dict of point estimates + (lo, hi) bands per setup.

Requires the JSON to contain `per_sample_toxic.{pred,gold}` arrays per setup
(added in the same commit that introduced this script).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


def _binary_metrics(pred: np.ndarray, gold: np.ndarray) -> Tuple[float, float, float]:
    tp = int(((pred == 1) & (gold == 1)).sum())
    fp = int(((pred == 1) & (gold == 0)).sum())
    fn = int(((pred == 0) & (gold == 1)).sum())
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f1


def bootstrap(pred: List[bool], gold: List[bool], n: int = 10_000, seed: int = 0) -> Dict[str, Dict[str, float]]:
    rng = np.random.default_rng(seed)
    pred_arr = np.array(pred, dtype=int)
    gold_arr = np.array(gold, dtype=int)
    N = len(pred_arr)
    p_pt, r_pt, f1_pt = _binary_metrics(pred_arr, gold_arr)

    samples = np.zeros((n, 3), dtype=float)
    for i in range(n):
        idx = rng.integers(0, N, size=N)
        samples[i] = _binary_metrics(pred_arr[idx], gold_arr[idx])
    lo, hi = np.percentile(samples, [2.5, 97.5], axis=0)
    return {
        "precision": {"point": p_pt, "ci_lo": float(lo[0]), "ci_hi": float(hi[0])},
        "recall":    {"point": r_pt, "ci_lo": float(lo[1]), "ci_hi": float(hi[1])},
        "f1":        {"point": f1_pt, "ci_lo": float(lo[2]), "ci_hi": float(hi[2])},
        "n":         N,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("input", help="TTP-Eval result JSON with per_sample_toxic arrays")
    ap.add_argument("--n", type=int, default=10_000, help="Bootstrap resamples (default 10000)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    payload = json.loads(Path(args.input).read_text())
    out: Dict[str, Dict] = {}
    for setup in payload.get("results", []):
        name = setup["setup"]
        ps = setup.get("per_sample_toxic")
        if not ps:
            print(f"[skip] {name}: no per_sample_toxic", file=sys.stderr)
            continue
        out[name] = bootstrap(ps["pred"], ps["gold"], n=args.n, seed=args.seed)

    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
