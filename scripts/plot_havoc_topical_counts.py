"""Generate HAVOC topical label frequency plot.

Outputs:
  - havoc-topical-counts.png

This reads the released `data/HAVOC/havoc.tsv` file directly (robust to extra tabs)
so that the plot can be regenerated consistently across environments.

Definition used here (aligns with the paper's Figure caption):
    - By default, a prompt is counted for a topical category if the corresponding
        entry in PrefixLab is exactly "topical".
    - If you want the broader "non-none" definition (includes "toxic"), pass
        --include-toxic.
"""

from __future__ import annotations

import argparse
import ast
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


# HAVOC PrefixLab has 5 dimensions aligned with the paper's taxonomy order:
# H (Hate & Violence), IH (Ideological Harm), SE (Sexual), IL (Illegal), SI (Self-Inflicted)
CATEGORIES = ["H", "IH", "SE", "IL", "SI"]
DISPLAY = ["H", "IH", "SE", "IL", "SI"]


def _parse_prefixlab(label_str: str) -> list[str] | None:
    label_str = (label_str or "").strip()
    if not label_str or label_str == "[]":
        return None
    try:
        labels: Any = ast.literal_eval(label_str)
    except Exception:
        return None
    if not isinstance(labels, list) or len(labels) != 5:
        return None
    return [str(item).strip().lower() for item in labels]


def _iter_havoc_rows(havoc_path: Path):
    with havoc_path.open("r", encoding="utf-8", errors="replace") as f:
        header = f.readline()
        if not header:
            return
        # Expect: Prefix\tSuffix\tPrefixLab
        for line_number, line in enumerate(f, start=2):
            line = (line or "").rstrip("\n")
            if not line:
                continue

            parts = line.split("\t")
            if len(parts) < 3:
                continue

            prefix = parts[0]
            prefix_lab_str = parts[-1]
            suffix = "\t".join(parts[1:-1])
            yield line_number, prefix, suffix, prefix_lab_str


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--havoc-tsv", default="data/HAVOC/havoc.tsv")
    parser.add_argument(
        "--out",
        default="havoc-topical-counts.png",
        help="Output PNG path (default: havoc-topical-counts.png)",
    )
    parser.add_argument(
        "--include-toxic",
        action="store_true",
        help="Count any non-'none' label (includes 'toxic') instead of only 'topical'.",
    )
    args = parser.parse_args()

    havoc_path = Path(args.havoc_tsv)
    if not havoc_path.exists():
        raise FileNotFoundError(f"HAVOC TSV not found: {havoc_path}")

    counts: Counter[str] = Counter()
    total_rows = 0
    parsed_rows = 0

    for _line_number, _prefix, _suffix, prefix_lab_str in _iter_havoc_rows(havoc_path):
        total_rows += 1
        labels = _parse_prefixlab(prefix_lab_str)
        if labels is None:
            continue
        parsed_rows += 1

        for category, label in zip(CATEGORIES, labels, strict=True):
            if args.include_toxic:
                if label != "none":
                    counts[category] += 1
            else:
                if label == "topical":
                    counts[category] += 1

    # Plot
    x = list(range(len(CATEGORIES)))
    y = [counts[c] for c in CATEGORIES]

    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    ax.bar(x, y, color="#4C78A8")
    ax.set_xticks(x, DISPLAY, rotation=0)
    ax.set_ylabel("Topical label count")
    ax.set_title("HAVOC topical label frequency")

    subtitle = f"rows parsed: {parsed_rows}/{total_rows}"
    ax.text(0.99, 0.98, subtitle, transform=ax.transAxes, ha="right", va="top", fontsize=9, alpha=0.8)

    fig.tight_layout()
    out_path = Path(args.out)
    fig.savefig(out_path, dpi=200)


if __name__ == "__main__":
    main()
