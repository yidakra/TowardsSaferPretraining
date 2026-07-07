#!/usr/bin/env python3
"""Per-document semantic-fidelity audit of the TTP-Eval translations.

Complements scripts/analyze_translation_degeneration.py: the repetition-loop
detector catches degenerate output, but NMT can also fail *fluently* by
hallucinating content absent from the source (see the French excerpt quoted in
review — "l'Université de Saint-Pierre-et-Loire ... le pays de l'Islam" has no
counterpart in the English prompt). Fluent hallucination is invisible to
loop/compression heuristics, so we measure it directly:

  1. embed each English source and its round-trip back-translation
     (results/back_translation/back_translated_<code>.tsv, NLLB-200-3.3B both
     directions) with a sentence encoder, chunk-averaged for long documents;
  2. per-document cosine similarity = round-trip semantic fidelity. Low
     fidelity is an UPPER bound on forward-translation hallucination (the
     back-translation leg adds its own noise);
  3. cross-tabulate with the repetition-loop flags: fluent hallucinations are
     low-fidelity docs that are NOT loop-flagged;
  4. recompute each classifier's per-language F1 on the STRICT-CLEAN subset
     (no loop AND fidelity >= threshold), paired against the same classifier's
     English F1 on the same rows.

    python scripts/audit_translation_fidelity.py
    python scripts/audit_translation_fidelity.py --threshold 0.6
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import zlib
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
csv.field_size_limit(10_000_000)

# back_translated_<code>.tsv -> translated TTPEval_<lang>.tsv naming
LANGS = {"es": "spa_Latn", "fr": "fra_Latn", "de": "deu_Latn",
         "ar": "arb_Arab", "hi": "hin_Deva", "zh": "zho_Hans"}

LOOP = re.compile(r"(.{2,15}?)\1{9,}")
ZLIB_FLOOR = 0.10
CHUNK_CHARS = 1500
MAX_CHUNKS = 4


def is_degenerate(text: str) -> bool:
    if not text:
        return True
    if LOOP.search(text):
        return True
    return len(zlib.compress(text.encode())) / max(len(text.encode()), 1) < ZLIB_FLOOR


def read_tsv(path: Path):
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def embed_docs(model, texts):
    """Chunk-averaged embeddings so long documents are not judged on their first 256 tokens."""
    flat, spans = [], []
    for t in texts:
        t = (t or "").strip()
        chunks = [t[i:i + CHUNK_CHARS] for i in range(0, len(t), CHUNK_CHARS)][:MAX_CHUNKS] or [""]
        spans.append((len(flat), len(flat) + len(chunks)))
        flat.extend(chunks)
    emb = model.encode(flat, batch_size=64, show_progress_bar=False, normalize_embeddings=True)
    out = np.vstack([emb[a:b].mean(axis=0) for a, b in spans])
    out /= np.clip(np.linalg.norm(out, axis=1, keepdims=True), 1e-9, None)
    return out


def f1_on(preds, golds, idx):
    tp = sum(1 for i in idx if preds[i] and golds[i])
    fp = sum(1 for i in idx if preds[i] and not golds[i])
    fn = sum(1 for i in idx if not preds[i] and golds[i])
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return 2 * p * r / (p + r) if p + r else 0.0


def per_sample(result_json: Path, setup: str):
    payload = json.loads(result_json.read_text())
    r = next(x for x in payload["results"] if x["setup"] == setup)
    return r["per_sample_toxic"]["pred"], r["per_sample_toxic"]["gold"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--threshold", type=float, default=0.5,
                    help="round-trip cosine below this = hallucination-suspect (default 0.5)")
    ap.add_argument("--encoder", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--output", default="results/translation_fidelity/fidelity_audit.json")
    args = ap.parse_args()

    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(args.encoder)

    english = read_tsv(ROOT / "data" / "TTP-Eval" / "TTPEval.tsv")
    url_to_idx = {r["URL"]: i for i, r in enumerate(english)}

    en_json = ROOT / "results/ttp_eval_baselines_with_preds/harmformer_llama_guard_en.json"
    setups = {"harmformer": "HarmFormer", "llama_guard": "Llama Guard"}
    en_ps = {k: per_sample(en_json, s) for k, s in setups.items()}

    report = {"threshold": args.threshold, "encoder": args.encoder, "languages": {}}
    print(f"Round-trip fidelity (cosine, {args.encoder}); hallucination-suspect < {args.threshold}")
    print(f"{'lang':<10}{'median':>8}{'p10':>7}{'suspect':>9}{'fluent-hall.':>13}  "
          f"{'HF strict':>10}{'EN@strict':>10}{'drop':>7}{'LG drop':>8}")
    for code, lang in sorted(LANGS.items(), key=lambda kv: kv[1]):
        bt = read_tsv(ROOT / "results" / "back_translation" / f"back_translated_{code}.tsv")
        # align to English row order via URL
        idx = [url_to_idx[r["URL"]] for r in bt]
        if sorted(idx) != list(range(len(english))):
            sys.exit(f"{lang}: back-translation rows do not cover the English set 1:1")
        src = [r["Source_English"] for r in bt]
        btx = [r["BT_Body"] for r in bt]
        e_src, e_bt = embed_docs(model, src), embed_docs(model, btx)
        cos_bt_order = (e_src * e_bt).sum(axis=1)
        cos = np.empty(len(english)); cos[idx] = cos_bt_order

        trans = read_tsv(ROOT / "data" / "TTP-Eval" / "translated" / "nllb-200-3.3B" / f"TTPEval_{lang}.tsv")
        degen = np.array([is_degenerate(r.get("Body", "")) for r in trans])
        suspect = cos < args.threshold
        fluent_hall = suspect & ~degen           # invisible to the loop detector
        strict = [i for i in range(len(english)) if not degen[i] and not suspect[i]]

        row = {"median_cos": float(np.median(cos)), "p10_cos": float(np.percentile(cos, 10)),
               "n_suspect": int(suspect.sum()), "n_fluent_hallucination": int(fluent_hall.sum()),
               "n_strict_clean": len(strict), "per_doc_cosine": [round(float(c), 4) for c in cos]}
        drops = {}
        for key, setup in setups.items():
            res_file = ROOT / "results/ttp_eval_multilingual" / f"{key}_{lang}.json"
            pred, gold = per_sample(res_file, setup)
            en_pred, en_gold = en_ps[key]
            assert gold == en_gold
            f_strict = f1_on(pred, gold, strict)
            f_en = f1_on(en_pred, en_gold, strict)
            drops[key] = (f_strict, f_en, f_strict - f_en)
            row[f"{key}_f1_strict"] = round(f_strict, 4)
            row[f"{key}_f1_en_same_rows"] = round(f_en, 4)
        report["languages"][lang] = row
        hf, lg = drops["harmformer"], drops["llama_guard"]
        print(f"{lang:<10}{np.median(cos):>8.3f}{np.percentile(cos,10):>7.3f}{suspect.sum():>9}"
              f"{fluent_hall.sum():>13}  {hf[0]:>10.3f}{hf[1]:>10.3f}{hf[2]:>7.2f}{lg[2]:>8.2f}")

    out = ROOT / args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(f"\nwrote {out}")
    print("Note: round-trip cosine lower-bounds forward fidelity (the back-translation leg adds noise),")
    print("so `suspect` over-counts hallucination; the strict-clean F1 is the conservative test.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
