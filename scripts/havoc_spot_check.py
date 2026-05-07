#!/usr/bin/env python3
"""HAVOC spot-check: independently re-judge a 200-row subset of released
Llama 3B continuations using HarmFormer and Llama Guard 3 8B as independent
classifiers, and compare verdicts against the released TTP/GPT-4o-judged
labels (Llama3BLab).

Addresses the reviewer concern that Claim 4 only verifies aggregation
arithmetic. Re-judging the released continuations with two local
classifiers (HarmFormer = same author family but different architecture;
Llama Guard 3 8B = independent of Mendu's pipeline) provides a
spot-check that the released judge labels are not artifacts of the
particular GPT-4o judging round.

Reads:
  - data/HAVOC/havoc_modeleval.tsv (released)

Outputs:
  - results/havoc_spotcheck/spotcheck_n200.json
  - results/havoc_spotcheck/spotcheck_n200_per_sample.tsv
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path
from typing import List, Tuple


def _read_modeleval_csv(path: Path):
    """Use proper csv.DictReader; the released file has CSV-quoted multiline cells."""
    with path.open("r", encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        return [dict(r) for r in reader]


def _toxic_from_lab(lab: str) -> int:
    """TTP taxonomy: a 5-tuple of {None, Topical-*, Intent-*}. The released
    HAVOC labels store these as lowercase 'none' / 'topical' / 'intent'.
    Per the paper, a continuation is Toxic iff any of the five elements is
    Intent (the published Claim 4 leakage definition); Topical alone is not
    toxic. We therefore detect 'intent' substrings, not 'toxic'."""
    if not lab:
        return 0
    s = lab.strip().lower()
    return 1 if "intent" in s else 0


def _kappa_2x2(a: List[int], b: List[int]) -> Tuple[float, float, dict]:
    """Cohen's kappa for two binary label vectors. Returns (agreement, kappa, confusion)."""
    n = len(a)
    if n == 0:
        return 0.0, 0.0, {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    matches = sum(1 for x, y in zip(a, b) if x == y)
    agr = matches / n
    tp = sum(1 for x, y in zip(a, b) if x == 1 and y == 1)
    fp = sum(1 for x, y in zip(a, b) if x == 1 and y == 0)
    fn = sum(1 for x, y in zip(a, b) if x == 0 and y == 1)
    tn = sum(1 for x, y in zip(a, b) if x == 0 and y == 0)
    p_e = (((tp + fp) / n) * ((tp + fn) / n)) + (((tn + fn) / n) * ((tn + fp) / n))
    kappa = (agr - p_e) / (1 - p_e) if (1 - p_e) > 1e-9 else 0.0
    return agr, kappa, {"tp": tp, "fp": fp, "fn": fn, "tn": tn}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--havoc-modeleval", default="data/HAVOC/havoc_modeleval.tsv")
    p.add_argument("--n", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--target-model", default="llama_3b",
                   help="Which released model column to spot-check (default: llama_3b)")
    p.add_argument("--out-dir", default="results/havoc_spotcheck")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    sys.path.insert(0, str(Path(__file__).parent.parent))
    from src.data_loaders.havoc_loader import MODEL_CONFIGS
    from src.models import HarmFormer
    from src.clients.llama_guard import LlamaGuardClient

    if args.target_model not in MODEL_CONFIGS:
        raise SystemExit(f"Unknown target-model {args.target_model}; choose from {list(MODEL_CONFIGS)}")
    response_col, judge_col, label_col = MODEL_CONFIGS[args.target_model]

    print(f"Loading {args.havoc_modeleval} ...", flush=True)
    rows = _read_modeleval_csv(Path(args.havoc_modeleval))
    print(f"Loaded {len(rows)} rows", flush=True)

    # Filter to rows with a non-empty released continuation and label.
    eligible = [r for r in rows if (r.get(response_col) or "").strip() and (r.get(label_col) or "").strip()]
    print(f"Eligible rows ({response_col} + {label_col} populated): {len(eligible)}", flush=True)

    # Stratified sample by released toxic vs non-toxic.
    pool_toxic = [r for r in eligible if _toxic_from_lab(r[label_col]) == 1]
    pool_safe = [r for r in eligible if _toxic_from_lab(r[label_col]) == 0]
    print(f"Pools: toxic={len(pool_toxic)} safe={len(pool_safe)}", flush=True)

    rng = random.Random(args.seed)
    n_toxic = min(args.n // 2, len(pool_toxic))
    n_safe = min(args.n - n_toxic, len(pool_safe))
    sample = rng.sample(pool_toxic, n_toxic) + rng.sample(pool_safe, n_safe)
    rng.shuffle(sample)
    print(f"Sampled n={len(sample)} (toxic={n_toxic}, safe={n_safe})", flush=True)

    continuations = [r[response_col] for r in sample]
    released_toxic = [_toxic_from_lab(r[label_col]) for r in sample]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Judge with HarmFormer.
    print("Loading HarmFormer ...", flush=True)
    t0 = time.time()
    hf = HarmFormer(device=args.device)
    print(f"HarmFormer loaded in {time.time()-t0:.1f}s", flush=True)

    t1 = time.time()
    hf_labels = hf.predict_batch(continuations, show_progress=False)
    hf_toxic = [int(lbl.is_toxic()) for lbl in hf_labels]
    hf_elapsed = time.time() - t1
    print(f"HarmFormer judged {len(hf_toxic)} in {hf_elapsed:.1f}s", flush=True)

    # Free HarmFormer before loading Llama Guard.
    del hf
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ---- Judge with Llama Guard 3 (focused prompt).
    print("Loading Llama Guard 3 8B ...", flush=True)
    t2 = time.time()
    lg = LlamaGuardClient(prompt_mode="focused", device=args.device)
    print(f"Llama Guard loaded in {time.time()-t2:.1f}s", flush=True)

    t3 = time.time()
    lg_toxic: List[int] = []
    for i, text in enumerate(continuations):
        try:
            lbl = lg.predict(text)
            lg_toxic.append(int(lbl.is_toxic()))
        except Exception as e:
            print(f"  [LG] sample {i} failed: {e}", flush=True)
            lg_toxic.append(0)  # fail-open
        if (i + 1) % 25 == 0:
            print(f"  LG judged {i+1}/{len(continuations)} ({time.time()-t3:.1f}s)", flush=True)
    lg_elapsed = time.time() - t3
    print(f"Llama Guard judged {len(lg_toxic)} in {lg_elapsed:.1f}s", flush=True)

    # ---- Agreement vs released labels.
    n = len(released_toxic)
    hf_agr, hf_kappa, hf_conf = _kappa_2x2(hf_toxic, released_toxic)
    lg_agr, lg_kappa, lg_conf = _kappa_2x2(lg_toxic, released_toxic)
    cross_agr, cross_kappa, _ = _kappa_2x2(hf_toxic, lg_toxic)

    summary = {
        "n": n,
        "target_model": args.target_model,
        "response_col": response_col,
        "label_col": label_col,
        "released_toxic_rate": round(sum(released_toxic) / n, 4) if n else None,
        "harmformer": {
            "judge": "themendu/HarmFormer (same author family, independent architecture)",
            "toxic_rate": round(sum(hf_toxic) / n, 4) if n else None,
            "agreement_vs_released": round(hf_agr, 4),
            "cohen_kappa": round(hf_kappa, 4),
            "confusion": hf_conf,
            "elapsed_s": round(hf_elapsed, 1),
        },
        "llama_guard_3": {
            "judge": "meta-llama/Llama-Guard-3-8B focused-prompt (independent pipeline)",
            "toxic_rate": round(sum(lg_toxic) / n, 4) if n else None,
            "agreement_vs_released": round(lg_agr, 4),
            "cohen_kappa": round(lg_kappa, 4),
            "confusion": lg_conf,
            "elapsed_s": round(lg_elapsed, 1),
        },
        "harmformer_vs_llama_guard": {
            "agreement": round(cross_agr, 4),
            "cohen_kappa": round(cross_kappa, 4),
        },
        "seed": args.seed,
    }
    print(json.dumps(summary, indent=2), flush=True)
    (out_dir / "spotcheck_n200.json").write_text(json.dumps(summary, indent=2))

    # Per-sample TSV.
    with (out_dir / "spotcheck_n200_per_sample.tsv").open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["idx", "released_label", "released_toxic", "hf_toxic", "lg_toxic",
                    "continuation_excerpt"])
        for i, (rel_lab, rt, ht, lt, cont) in enumerate(zip(
            [r[label_col] for r in sample], released_toxic, hf_toxic, lg_toxic, continuations
        )):
            w.writerow([i, rel_lab, rt, ht, lt, (cont or "")[:300].replace("\t", " ").replace("\n", " ")])
    print(f"Wrote {out_dir}/spotcheck_n200.json and per_sample.tsv", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
