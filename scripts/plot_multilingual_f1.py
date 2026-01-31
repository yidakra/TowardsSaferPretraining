"""Generate the multilingual F1 figure used in the paper.

Outputs:
  - multilingual-f1.png

Inputs:
  - results/ttp_eval_multilingual/harmformer_*.json
  - results/ttp_eval_multilingual/llama_guard_*.json

Optional English reference bars:
  - HarmFormer: read from results/harmformer/harmformer_results.json (TTP-Eval)
    - Llama Guard: best-effort, from one of:
            * --llama-guard-en-f1
            * --llama-guard-en-json (English TTP-Eval run)
            * --llama-guard-en-table7-json (Table 7 OpenAI Moderation results)

This script is intentionally lightweight and does not re-run model inference.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


LANG_DISPLAY = {
    "spa_Latn": "Spanish",
    "fra_Latn": "French",
    "deu_Latn": "German",
    "arb_Arab": "Arabic",
    "hin_Deva": "Hindi",
    "zho_Hans": "Chinese",
}

LANG_ORDER = ["English", "Spanish", "French", "German", "Arabic", "Hindi", "Chinese"]


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _extract_overall_f1(payload: Dict[str, Any]) -> float:
    # multilingual schema: {"results": [{"metrics": {"overall": {"f1": ...}}}]}
    results = payload.get("results")
    if isinstance(results, list) and results:
        metrics = results[0].get("metrics") if isinstance(results[0], dict) else None
        if isinstance(metrics, dict):
            overall = metrics.get("overall")
            if isinstance(overall, dict) and "f1" in overall:
                return float(overall["f1"])

    # harmformer schema: {"metrics": {"overall": {"f1": ...}}}
    metrics = payload.get("metrics")
    if isinstance(metrics, dict):
        overall = metrics.get("overall")
        if isinstance(overall, dict) and "f1" in overall:
            return float(overall["f1"])

    raise KeyError("Could not find overall.f1 in payload")


def _parse_lang_from_filename(path: Path, prefix: str) -> Optional[str]:
    # e.g. harmformer_spa_Latn.json -> spa_Latn
    m = re.match(rf"^{re.escape(prefix)}_(.+)\.json$", path.name)
    if not m:
        return None
    return m.group(1)


def load_multilingual_dir(results_dir: Path, prefix: str) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for p in sorted(results_dir.glob(f"{prefix}_*.json")):
        lang_code = _parse_lang_from_filename(p, prefix)
        if not lang_code:
            continue
        display = LANG_DISPLAY.get(lang_code)
        if not display:
            continue
        out[display] = _extract_overall_f1(_read_json(p))
    return out


def _extract_table7_llama_guard_f1(payload: Dict[str, Any]) -> Optional[float]:
    # Table 7 schema: {"results": [{"classifier": "Llama Guard", "metrics": {"overall": {"f1": ...}}}, ...]}
    results = payload.get("results")
    if not isinstance(results, list):
        return None
    for item in results:
        if not isinstance(item, dict):
            continue
        if item.get("classifier") != "Llama Guard":
            continue
        metrics = item.get("metrics")
        if not isinstance(metrics, dict):
            return None
        overall = metrics.get("overall")
        if not isinstance(overall, dict):
            return None
        if "f1" not in overall:
            return None
        try:
            return float(overall["f1"])
        except Exception:
            return None
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--multilingual-results-dir", default="results/ttp_eval_multilingual")
    parser.add_argument("--harmformer-en-json", default="results/harmformer/harmformer_results.json")
    parser.add_argument(
        "--llama-guard-en-json",
        default="results/ttp_eval_baselines/llama_guard_en.json",
        help=(
            "Optional JSON file containing an English Llama Guard run on TTP-Eval. "
            "If present and --llama-guard-en-f1 is not provided, we will extract overall.f1 from it."
        ),
    )
    parser.add_argument(
        "--llama-guard-en-f1",
        type=float,
        default=None,
        help="Optional English F1 for Llama Guard on (English) TTP-Eval.",
    )
    parser.add_argument(
        "--llama-guard-en-table7-json",
        default="results/moderation/table7_local_results.json",
        help=(
            "Optional Table 7 (OpenAI Moderation) JSON file. If provided and the English Llama Guard "
            "TTP-Eval JSON is missing, we will use the 'Llama Guard' overall.f1 from this file as the "
            "English reference bar."
        ),
    )
    parser.add_argument("--out", default="multilingual-f1.png")
    args = parser.parse_args()

    multilingual_dir = Path(args.multilingual_results_dir)
    harmformer_map = load_multilingual_dir(multilingual_dir, "harmformer")
    llama_guard_map = load_multilingual_dir(multilingual_dir, "llama_guard")

    # English reference (best-effort)
    harmformer_en: Optional[float] = None
    harmformer_en_path = Path(args.harmformer_en_json)
    if harmformer_en_path.exists():
        harmformer_en = _extract_overall_f1(_read_json(harmformer_en_path))

    llama_guard_en: Optional[float] = args.llama_guard_en_f1
    if llama_guard_en is None:
        llama_guard_en_path = Path(args.llama_guard_en_json)
        if llama_guard_en_path.exists():
            try:
                llama_guard_en = _extract_overall_f1(_read_json(llama_guard_en_path))
            except Exception:
                llama_guard_en = None

    if llama_guard_en is None:
        table7_path = Path(args.llama_guard_en_table7_json)
        if table7_path.exists():
            try:
                llama_guard_en = _extract_table7_llama_guard_f1(_read_json(table7_path))
            except Exception:
                llama_guard_en = None

    # Build aligned series
    harmformer_series: List[Optional[float]] = []
    llama_guard_series: List[Optional[float]] = []

    for lang in LANG_ORDER:
        if lang == "English":
            harmformer_series.append(harmformer_en)
            llama_guard_series.append(llama_guard_en)
        else:
            harmformer_series.append(harmformer_map.get(lang))
            llama_guard_series.append(llama_guard_map.get(lang))

    # Plot
    x = list(range(len(LANG_ORDER)))
    width = 0.38

    fig, ax = plt.subplots(figsize=(8.2, 4.2))

    def _bar_positions(offset: float) -> List[float]:
        return [i + offset for i in x]

    # Replace missing values with NaN so matplotlib skips annotations cleanly.
    def _to_float_or_nan(v: Optional[float]) -> float:
        return float("nan") if v is None else float(v)

    harmformer_vals = [_to_float_or_nan(v) for v in harmformer_series]
    llama_guard_vals = [_to_float_or_nan(v) for v in llama_guard_series]

    ax.bar(_bar_positions(-width / 2), harmformer_vals, width=width, label="HarmFormer", color="#4C78A8")
    ax.bar(_bar_positions(+width / 2), llama_guard_vals, width=width, label="Llama Guard", color="#F58518")

    ax.set_xticks(x, LANG_ORDER, rotation=0)
    ax.set_ylabel("Overall toxic F1")
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Multilingual TTP-Eval (translated) — toxic F1 by language")
    ax.legend(frameon=False)

    note_bits: List[str] = []
    if harmformer_en is None:
        note_bits.append("HarmFormer English: missing")
    if llama_guard_en is None:
        note_bits.append("Llama Guard English: missing")
    if note_bits:
        ax.text(0.99, 0.02, "; ".join(note_bits), transform=ax.transAxes, ha="right", va="bottom", fontsize=9, alpha=0.8)

    # Annotate non-NaN bars
    for i, (h, l) in enumerate(zip(harmformer_series, llama_guard_series, strict=True)):
        if h is not None:
            ax.text(i - width / 2, h + 0.015, f"{h:.2f}", ha="center", va="bottom", fontsize=9)
        if l is not None:
            ax.text(i + width / 2, l + 0.015, f"{l:.2f}", ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    fig.savefig(Path(args.out), dpi=200)


if __name__ == "__main__":
    main()
