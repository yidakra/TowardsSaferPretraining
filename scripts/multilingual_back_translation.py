#!/usr/bin/env python3
"""Back-translate the NLLB-3.3B translated TTP-Eval back to English and report
BLEU and chrF against the original English source as a translation-noise floor.

Reads the existing translated TSVs under
data/TTP-Eval/translated/nllb-200-3.3B/, back-translates the Body column to
English with the same NLLB-200-3.3B model, and aligns rows with
data/TTP-Eval/TTPEval.tsv by URL. Reports corpus-level BLEU and chrF per
language.

Output: results/back_translation/back_translation_metrics.json
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List

LANG_TO_NLLB = {
    "es": "spa_Latn",
    "fr": "fra_Latn",
    "de": "deu_Latn",
    "ar": "arb_Arab",
    "hi": "hin_Deva",
    "zh": "zho_Hans",
}


def _read_tsv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        return [dict(r) for r in reader]


def _batched(items: List[str], batch_size: int) -> Iterable[List[str]]:
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--source-tsv", default="data/TTP-Eval/TTPEval.tsv")
    p.add_argument("--translated-dir", default="data/TTP-Eval/translated/nllb-200-3.3B")
    p.add_argument("--model-id", default="facebook/nllb-200-3.3B")
    p.add_argument("--out", default="results/back_translation/back_translation_metrics.json")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    import sacrebleu
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    src_rows = _read_tsv(Path(args.source_tsv))
    src_by_url = {r["URL"]: r["Body"] for r in src_rows}
    print(f"Loaded {len(src_rows)} source rows", flush=True)

    print(f"Loading model {args.model_id} ...", flush=True)
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(args.model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_id)
    if args.device == "cuda" and torch.cuda.is_available():
        model = model.to("cuda")
    model.eval()
    print(f"Loaded in {time.time()-t0:.1f}s", flush=True)

    eng_bos = tok.convert_tokens_to_ids("eng_Latn")
    if eng_bos == tok.unk_token_id:
        raise RuntimeError("eng_Latn token not in tokenizer vocabulary")

    results: Dict[str, Dict[str, float]] = {}
    out_dir = Path(args.out).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    for short, nllb_code in LANG_TO_NLLB.items():
        src_path = Path(args.translated_dir) / f"TTPEval_{nllb_code}.tsv"
        if not src_path.exists():
            print(f"[skip] {src_path} not found", flush=True)
            continue

        rows = _read_tsv(src_path)
        # Align rows with the English source by URL.
        refs: List[str] = []
        srcs: List[str] = []
        urls: List[str] = []
        for r in rows:
            url = r["URL"]
            if url not in src_by_url:
                continue
            refs.append(src_by_url[url])
            srcs.append(r["Body"])
            urls.append(url)

        print(f"[{short}] {len(srcs)} rows aligned (out of {len(rows)})", flush=True)
        tok.src_lang = nllb_code
        bt_outputs: List[str] = []
        t1 = time.time()
        for batch in _batched(srcs, args.batch_size):
            inputs = tok(batch, return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            with torch.no_grad():
                gen = model.generate(
                    **inputs,
                    forced_bos_token_id=eng_bos,
                    max_new_tokens=args.max_new_tokens,
                )
            bt_outputs.extend(tok.batch_decode(gen, skip_special_tokens=True))
        elapsed = time.time() - t1
        print(f"[{short}] back-translated {len(bt_outputs)} rows in {elapsed:.1f}s", flush=True)

        # Save back-translation TSV for inspection.
        bt_tsv = out_dir / f"back_translated_{short}.tsv"
        with bt_tsv.open("w", encoding="utf-8", newline="") as f:
            w = csv.writer(f, delimiter="\t")
            w.writerow(["URL", "BT_Body", "Source_English"])
            for u, bt, ref in zip(urls, bt_outputs, refs):
                w.writerow([u, bt, ref])

        # Corpus BLEU and chrF.
        bleu = sacrebleu.corpus_bleu(bt_outputs, [refs])
        chrf = sacrebleu.corpus_chrf(bt_outputs, [refs])
        results[short] = {
            "n_aligned": len(refs),
            "bleu": round(bleu.score, 2),
            "chrf": round(chrf.score, 2),
            "elapsed_s": round(elapsed, 1),
            "bt_tsv": str(bt_tsv),
        }
        print(f"[{short}] BLEU={bleu.score:.2f}  chrF={chrf.score:.2f}", flush=True)

    out_path = Path(args.out)
    out_path.write_text(json.dumps({"per_language": results, "model": args.model_id}, indent=2))
    print(f"Saved {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
