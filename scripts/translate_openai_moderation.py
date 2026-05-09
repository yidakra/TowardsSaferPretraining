"""Translate OpenAI Moderation samples-1680.jsonl.gz prompts via NLLB-200.

Mirrors `scripts/translate_ttp_eval.py` but for the OpenAI Moderation JSONL
schema (text in `prompt` field, binary harm flags in S/H/V/HR/SH/S3/H2/V2).
The binary flags are preserved as-is so the existing OAI Mod loader and
evaluator (`scripts/evaluate_openai_moderation.py`) work on the translated
files without modification.

Usage:
    python scripts/translate_openai_moderation.py \\
        --tgt-langs spa_Latn fra_Latn \\
        --output-dir data/moderation-api-release/translated/nllb-200-3.3B
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from pathlib import Path
from typing import Iterable, List

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.utils.codecarbon import maybe_track_emissions


def _batched(items: List[str], batch_size: int) -> Iterable[List[str]]:
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


def main():
    p = argparse.ArgumentParser(description="Translate samples-1680.jsonl.gz prompts using NLLB")
    p.add_argument("--input", default="data/moderation-api-release/data/samples-1680.jsonl.gz")
    p.add_argument("--src-lang", default="eng_Latn")
    p.add_argument("--tgt-langs", nargs="+", required=True, help="NLLB target language codes")
    p.add_argument("--model-id", default="facebook/nllb-200-3.3B")
    p.add_argument("--output-dir", default="data/moderation-api-release/translated/nllb-200-3.3B")
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu", "mps"])
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--limit", type=int, default=None, help="Optional row limit for quick tests")
    args = p.parse_args()

    in_path = Path(args.input)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    open_fn = gzip.open if in_path.suffix == ".gz" else open
    with open_fn(in_path, "rt", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    if not rows:
        raise ValueError(f"No rows in {in_path}")

    if args.limit is not None:
        rows = rows[: max(0, int(args.limit))]
    print(f"Loaded {len(rows)} rows from {in_path}", flush=True)

    model_slug = args.model_id.replace("/", "_")
    with maybe_track_emissions(run_name=f"translate_openai_moderation_{model_slug}"):
        tokenizer = AutoTokenizer.from_pretrained(args.model_id)
        model = AutoModelForSeq2SeqLM.from_pretrained(args.model_id)
        if args.device == "cuda" and torch.cuda.is_available():
            model = model.to("cuda")
        elif args.device == "mps" and torch.backends.mps.is_available():
            model = model.to("mps")

        tokenizer.src_lang = args.src_lang
        source_texts = [(r.get("prompt") or "") for r in rows]

        for tgt_lang in args.tgt_langs:
            forced_bos = tokenizer.convert_tokens_to_ids(tgt_lang)
            if forced_bos is None or forced_bos == tokenizer.unk_token_id:
                raise ValueError(f"Unknown tgt lang for tokenizer: {tgt_lang}")

            translated: List[str] = []
            for batch in _batched(source_texts, args.batch_size):
                inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=512)
                inputs = {k: v.to(model.device) for k, v in inputs.items()}
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        forced_bos_token_id=forced_bos,
                        max_new_tokens=args.max_new_tokens,
                    )
                translated.extend(tokenizer.batch_decode(outputs, skip_special_tokens=True))
            if len(translated) != len(rows):
                raise RuntimeError(f"Translation length mismatch: {len(translated)} vs {len(rows)}")

            out_path = out_dir / f"samples-1680_{tgt_lang}.jsonl.gz"
            with gzip.open(out_path, "wt", encoding="utf-8") as f:
                for r, t in zip(rows, translated, strict=True):
                    new_row = dict(r)
                    new_row["prompt"] = t
                    new_row["_src_lang"] = args.src_lang
                    new_row["_tgt_lang"] = tgt_lang
                    f.write(json.dumps(new_row, ensure_ascii=False) + "\n")
            print(f"Wrote: {out_path}", flush=True)


if __name__ == "__main__":
    main()
