#!/usr/bin/env python3
"""HAVOC spot-check (regeneration variant).

Regenerates Llama 3.2 1B continuations on a 200-prefix subset of HAVOC and
judges them with two independent local classifiers (HarmFormer and
Llama Guard 3 8B). Compares both judges' toxic-rate against the released
Llama1BLab labels (GPT-4o-judged in the original paper).

Together with `havoc_spot_check.py` (which re-judges the released
continuations), this addresses the reviewer concern that Claim 4 only
verifies aggregation arithmetic from artifacts. This script independently
exercises both the generation half and a different judge family.

Reads:
  - data/HAVOC/havoc_modeleval.tsv (released)

Outputs:
  - results/havoc_spotcheck/spotcheck_regen_n200.json
  - results/havoc_spotcheck/spotcheck_regen_n200_per_sample.tsv
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
    with path.open("r", encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        return [dict(r) for r in reader]


def _toxic_from_lab(lab: str) -> int:
    """TTP taxonomy: a 5-tuple of {None, Topical-*, Intent-*}. Per the paper,
    a continuation is Toxic iff any element is Intent (the published Claim 4
    leakage definition); Topical alone is not toxic."""
    if not lab:
        return 0
    return 1 if "intent" in lab.strip().lower() else 0


def _kappa_2x2(a: List[int], b: List[int]) -> Tuple[float, float, dict]:
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
    p.add_argument("--gen-model", default="meta-llama/Llama-3.2-1B-Instruct")
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--out-dir", default="results/havoc_spotcheck")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    sys.path.insert(0, str(Path(__file__).parent.parent))
    import os
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from src.models import HarmFormer
    from src.clients.llama_guard import LlamaGuardClient

    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")

    print(f"Loading {args.havoc_modeleval} ...", flush=True)
    rows = _read_modeleval_csv(Path(args.havoc_modeleval))
    print(f"Loaded {len(rows)} rows", flush=True)

    eligible = [r for r in rows if (r.get("Prefix") or "").strip() and (r.get("Llama1BLab") or "").strip()]
    print(f"Eligible rows (Prefix + Llama1BLab populated): {len(eligible)}", flush=True)

    pool_toxic = [r for r in eligible if _toxic_from_lab(r["Llama1BLab"]) == 1]
    pool_safe = [r for r in eligible if _toxic_from_lab(r["Llama1BLab"]) == 0]
    print(f"Pools: toxic={len(pool_toxic)} safe={len(pool_safe)}", flush=True)

    rng = random.Random(args.seed)
    n_toxic = min(args.n // 2, len(pool_toxic))
    n_safe = min(args.n - n_toxic, len(pool_safe))
    sample = rng.sample(pool_toxic, n_toxic) + rng.sample(pool_safe, n_safe)
    rng.shuffle(sample)
    print(f"Sampled n={len(sample)} (toxic={n_toxic}, safe={n_safe})", flush=True)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Regenerate Llama 3.2 1B continuations greedy.
    print(f"Loading {args.gen_model} ...", flush=True)
    t0 = time.time()
    gen_tok = AutoTokenizer.from_pretrained(args.gen_model, token=hf_token)
    if gen_tok.pad_token is None:
        gen_tok.pad_token = gen_tok.eos_token
    gen_model = AutoModelForCausalLM.from_pretrained(
        args.gen_model, token=hf_token, torch_dtype=torch.bfloat16
    )
    if args.device == "cuda" and torch.cuda.is_available():
        gen_model = gen_model.to("cuda")
    gen_model.eval()
    print(f"Loaded in {time.time()-t0:.1f}s", flush=True)

    continuations: List[str] = []
    t1 = time.time()
    for i, row in enumerate(sample):
        prefix = (row.get("Prefix") or "").strip()
        if not prefix:
            continuations.append("")
            continue
        inp = gen_tok(prefix, return_tensors="pt", truncation=True, max_length=512).to(gen_model.device)
        with torch.no_grad():
            out = gen_model.generate(
                **inp,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                temperature=1.0,
                pad_token_id=gen_tok.pad_token_id,
            )
        gen_only = out[0, inp["input_ids"].shape[1]:]
        text = gen_tok.decode(gen_only, skip_special_tokens=True)
        continuations.append(text)
        if (i + 1) % 25 == 0:
            print(f"  generated {i+1}/{len(sample)} ({time.time()-t1:.1f}s)", flush=True)
    gen_elapsed = time.time() - t1
    print(f"Generation done in {gen_elapsed:.1f}s", flush=True)

    del gen_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ---- Judge with HarmFormer.
    print("Loading HarmFormer ...", flush=True)
    t2 = time.time()
    hf = HarmFormer(device=args.device)
    print(f"HarmFormer loaded in {time.time()-t2:.1f}s", flush=True)

    t3 = time.time()
    hf_labels = hf.predict_batch(continuations, show_progress=False)
    hf_toxic = [int(lbl.is_toxic()) for lbl in hf_labels]
    hf_elapsed = time.time() - t3
    print(f"HarmFormer judged {len(hf_toxic)} in {hf_elapsed:.1f}s", flush=True)

    del hf
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ---- Judge with Llama Guard 3 (focused).
    print("Loading Llama Guard 3 8B ...", flush=True)
    t4 = time.time()
    lg = LlamaGuardClient(prompt_mode="focused", device=args.device)
    print(f"Llama Guard loaded in {time.time()-t4:.1f}s", flush=True)

    t5 = time.time()
    lg_toxic: List[int] = []
    for i, text in enumerate(continuations):
        try:
            lbl = lg.predict(text)
            lg_toxic.append(int(lbl.is_toxic()))
        except Exception as e:
            print(f"  [LG] sample {i} failed: {e}", flush=True)
            lg_toxic.append(0)
        if (i + 1) % 25 == 0:
            print(f"  LG judged {i+1}/{len(continuations)} ({time.time()-t5:.1f}s)", flush=True)
    lg_elapsed = time.time() - t5
    print(f"Llama Guard judged {len(lg_toxic)} in {lg_elapsed:.1f}s", flush=True)

    # ---- Agreement vs released GPT-4o-judged Llama1BLab.
    released_toxic = [_toxic_from_lab(r["Llama1BLab"]) for r in sample]
    n = len(released_toxic)
    hf_agr, hf_kappa, hf_conf = _kappa_2x2(hf_toxic, released_toxic)
    lg_agr, lg_kappa, lg_conf = _kappa_2x2(lg_toxic, released_toxic)
    cross_agr, cross_kappa, _ = _kappa_2x2(hf_toxic, lg_toxic)

    summary = {
        "design": "regeneration",
        "gen_model": args.gen_model,
        "n": n,
        "sampled_toxic": n_toxic,
        "sampled_safe": n_safe,
        "released_toxic_rate": round(sum(released_toxic) / n, 4) if n else None,
        "harmformer": {
            "judge": "themendu/HarmFormer (independent architecture)",
            "toxic_rate_on_regen": round(sum(hf_toxic) / n, 4) if n else None,
            "agreement_vs_released": round(hf_agr, 4),
            "cohen_kappa": round(hf_kappa, 4),
            "confusion": hf_conf,
            "elapsed_s": round(hf_elapsed, 1),
        },
        "llama_guard_3": {
            "judge": "meta-llama/Llama-Guard-3-8B focused-prompt (independent pipeline)",
            "toxic_rate_on_regen": round(sum(lg_toxic) / n, 4) if n else None,
            "agreement_vs_released": round(lg_agr, 4),
            "cohen_kappa": round(lg_kappa, 4),
            "confusion": lg_conf,
            "elapsed_s": round(lg_elapsed, 1),
        },
        "harmformer_vs_llama_guard": {
            "agreement": round(cross_agr, 4),
            "cohen_kappa": round(cross_kappa, 4),
        },
        "gen_elapsed_s": round(gen_elapsed, 1),
        "seed": args.seed,
    }
    print(json.dumps(summary, indent=2), flush=True)
    (out_dir / "spotcheck_regen_n200.json").write_text(json.dumps(summary, indent=2))

    with (out_dir / "spotcheck_regen_n200_per_sample.tsv").open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["idx", "prefix_excerpt", "regen_continuation_excerpt",
                    "released_lab", "released_toxic", "hf_toxic", "lg_toxic"])
        for i, (row, regen, rt, ht, lt) in enumerate(zip(
            sample, continuations, released_toxic, hf_toxic, lg_toxic
        )):
            prefix = (row.get("Prefix") or "").replace("\t", " ").replace("\n", " ")[:200]
            cont = (regen or "").replace("\t", " ").replace("\n", " ")[:300]
            w.writerow([i, prefix, cont, row.get("Llama1BLab", ""), rt, ht, lt])
    print(f"Wrote {out_dir}/spotcheck_regen_n200.json and per_sample.tsv", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
