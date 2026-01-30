#!/usr/bin/env python3
"""Translate TTP-Eval (TTPEval.tsv) into multiple languages using NLLB.

This is provided to support the multilingual *extension* reproduction.

Notes:
- Translation is expensive (model downloads + GPU strongly recommended).
- The repo already ships translated TSVs under `data/TTP-Eval/translated/` for convenience.
- This script regenerates those artifacts from `data/TTP-Eval/TTPEval.tsv`.

Outputs are TSVs named `TTPEval_<tgt_lang>.tsv` in the chosen output directory.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, Iterable, List


def _read_tsv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        return [dict(r) for r in reader]


def _write_tsv(path: Path, rows: List[Dict[str, str]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, delimiter="\t", fieldnames=fieldnames, quoting=csv.QUOTE_MINIMAL)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def _batched(items: List[str], batch_size: int) -> Iterable[List[str]]:
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


def _lang_cell_from_nllb_code(nllb_code: str) -> str:
    # NLLB codes look like `spa_Latn`. The existing shipped translations store `spa`.
    return nllb_code.split("_", 1)[0]


def main() -> int:
    p = argparse.ArgumentParser(description="Translate TTPEval.tsv using NLLB")
    p.add_argument("--input", default="data/TTP-Eval/TTPEval.tsv")
    p.add_argument(
        "--model-id",
        default="facebook/nllb-200-distilled-600M",
        help="NLLB model to use (e.g., facebook/nllb-200-3.3B)",
    )
    p.add_argument(
        "--src-lang",
        default="eng_Latn",
        help="NLLB source language code for the input (default: eng_Latn)",
    )
    p.add_argument(
        "--tgt-langs",
        nargs="+",
        required=True,
        help="One or more NLLB target language codes (e.g., spa_Latn deu_Latn arb_Arab)",
    )
    p.add_argument("--limit", type=int, default=None, help="Optional row limit for quick tests")
    p.add_argument(
        "--output-dir",
        default=None,
        help="Output dir. Default: data/TTP-Eval/translated/<model-name>/",
    )
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu", "mps"])
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-new-tokens", type=int, default=256)
    args = p.parse_args()

    # Defer heavy imports so `--help` is fast.
    import torch  # noqa: WPS433
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer  # noqa: WPS433

    in_path = Path(args.input)
    if not in_path.exists():
        raise FileNotFoundError(in_path)

    model_name_slug = args.model_id.split("/", 1)[-1]
    out_dir = Path(args.output_dir) if args.output_dir else Path("data/TTP-Eval/translated") / model_name_slug

    rows = _read_tsv(in_path)
    if not rows:
        raise ValueError(f"No rows in {in_path}")

    if args.limit is not None:
        rows = rows[: max(0, int(args.limit))]
        if not rows:
            raise ValueError("--limit resulted in 0 rows")

    fieldnames = list(rows[0].keys())
    if "Body" not in fieldnames or "Lang" not in fieldnames:
        raise ValueError("Expected columns include at least 'Body' and 'Lang'")

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_id)

    if args.device == "cuda" and torch.cuda.is_available():
        model = model.to("cuda")
    elif args.device == "mps" and torch.backends.mps.is_available():
        model = model.to("mps")
    else:
        model = model.to("cpu")

    tokenizer.src_lang = args.src_lang

    source_texts = [r.get("Body", "") for r in rows]

    for tgt_lang in args.tgt_langs:
        forced_bos_token_id = tokenizer.convert_tokens_to_ids(tgt_lang)
        if forced_bos_token_id is None:
            raise ValueError(f"Unknown tgt lang for tokenizer: {tgt_lang}")

        translated: List[str] = []
        for batch in _batched(source_texts, args.batch_size):
            inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    forced_bos_token_id=forced_bos_token_id,
                    max_new_tokens=args.max_new_tokens,
                )
            translated.extend(tokenizer.batch_decode(outputs, skip_special_tokens=True))

        if len(translated) != len(rows):
            raise RuntimeError("Translation batch size mismatch")

        out_rows: List[Dict[str, str]] = []
        lang_cell = _lang_cell_from_nllb_code(tgt_lang)
        for r, t in zip(rows, translated, strict=True):
            rr = dict(r)
            rr["Body"] = t
            rr["Lang"] = lang_cell
            out_rows.append(rr)

        out_path = out_dir / f"TTPEval_{tgt_lang}.tsv"
        _write_tsv(out_path, out_rows, fieldnames=fieldnames)
        print(f"Wrote: {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
