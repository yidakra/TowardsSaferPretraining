#!/usr/bin/env python3
"""
Compute non-safe (topical+toxic) harm percentages for HAVOC and RTP
and plot a radar chart.

RTP labels are obtained by running HarmFormer on prompt+continuation.
HAVOC labels are taken from the provided topical annotations.
"""
from __future__ import annotations

import argparse
import ast
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, cast

import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from datasets import load_dataset  # type: ignore

from src.models import HarmFormer
from src.utils.taxonomy import HarmLabel, Dimension

CATS = ["Hate & Violence", "Ideological Harm", "Sexual", "Illegal Behavior", "Self-Inflicted Harm"]
CAT_ATTRS = ["hate_violence", "ideological", "sexual", "illegal", "self_inflicted"]
CAT_SHORT = ["H", "IH", "SE", "IL", "SI"]


def _count_havoc(havoc_path: Path) -> Dict[str, float]:
    total = 0
    counts = Counter({c: 0 for c in CAT_SHORT})
    with havoc_path.open("r", encoding="utf-8") as f:
        _ = f.readline()  # header
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            try:
                raw_labels = ast.literal_eval(parts[2])
            except Exception:
                continue
            if not isinstance(raw_labels, list):
                continue
            raw_labels_any = cast(List[Any], raw_labels)
            if len(raw_labels_any) != 5:
                continue
            labels = [str(x) for x in raw_labels_any]
            total += 1
            for short, value in zip(CAT_SHORT, labels):
                if str(value).lower() != "none":
                    counts[short] += 1
    return {short: (counts[short] / total * 100.0) if total else 0.0 for short in CAT_SHORT}


def _count_rtp(device: str, batch_size: int, limit: int | None) -> Dict[str, float]:
    judge = HarmFormer(device=device, batch_size=batch_size)
    dataset = cast(
        Iterable[Mapping[str, Any]],
        load_dataset("allenai/real-toxicity-prompts", split="train", streaming=True),
    )

    total = 0
    counts = Counter({attr: 0 for attr in CAT_ATTRS})
    batch: List[str] = []

    def flush(batch_texts: List[str]) -> None:
        nonlocal total
        if not batch_texts:
            return
        labels: List[HarmLabel] = judge.predict_batch(batch_texts, show_progress=False)
        for label in labels:
            total += 1
            for attr in CAT_ATTRS:
                if getattr(label, attr) != Dimension.SAFE:
                    counts[attr] += 1

    def _get_text(row: Mapping[str, Any], key: str) -> str:
        value = row.get(key)
        if isinstance(value, Mapping):
            value_map = cast(Mapping[str, Any], value)
            text_value = value_map.get("text")
            return str(text_value) if text_value is not None else ""
        return ""

    for idx, row in enumerate(dataset):
        if limit is not None and idx >= limit:
            break
        prompt = _get_text(row, "prompt")
        continuation = _get_text(row, "continuation")
        text = (prompt + " " + continuation).strip()
        if not text:
            continue
        batch.append(text)
        if len(batch) >= batch_size:
            flush(batch)
            batch = []

    flush(batch)

    return {
        "H": (counts["hate_violence"] / total * 100.0) if total else 0.0,
        "IH": (counts["ideological"] / total * 100.0) if total else 0.0,
        "SE": (counts["sexual"] / total * 100.0) if total else 0.0,
        "IL": (counts["illegal"] / total * 100.0) if total else 0.0,
        "SI": (counts["self_inflicted"] / total * 100.0) if total else 0.0,
    }


def _plot_radar(havoc: Dict[str, float], rtp: Dict[str, float], out_path: Path) -> None:
    import math
    import matplotlib.pyplot as plt

    values_havoc = [havoc[k] for k in CAT_SHORT]
    values_rtp = [rtp[k] for k in CAT_SHORT]

    angles = [n / float(len(CATS)) * 2 * math.pi for n in range(len(CATS))]
    angles += angles[:1]

    values_havoc += values_havoc[:1]
    values_rtp += values_rtp[:1]

    fig, ax = plt.subplots(figsize=(6, 4), subplot_kw=dict(polar=True))  # type: ignore
    fig = cast(Any, fig)
    ax = cast(Any, ax)
    ax.set_theta_offset(math.pi / 2)
    ax.set_theta_direction(-1)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(CATS)

    ax.plot(angles, values_havoc, color="#1f77b4", linewidth=2, label="HAVOC")
    ax.fill(angles, values_havoc, color="#1f77b4", alpha=0.2)

    ax.plot(angles, values_rtp, color="#ff7f0e", linewidth=2, linestyle="--", label="RTP")
    ax.fill(angles, values_rtp, color="#ff7f0e", alpha=0.15)

    ax.set_ylim(0, max(max(values_havoc), max(values_rtp)) * 1.1)
    ax.set_yticklabels([])

    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 1.15))
    plt.tight_layout()
    fig.savefig(out_path, dpi=200)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--havoc-path", default="data/HAVOC/havoc.tsv")
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--limit", type=int, help="Optional RTP sample limit")
    p.add_argument("--out-json", default="results/rtp/havoc_rtp_harms.json")
    p.add_argument("--out-fig", default="havoc-rtp-harms-radar.png")
    args = p.parse_args()

    havoc = _count_havoc(Path(args.havoc_path))
    rtp = _count_rtp(device=args.device, batch_size=args.batch_size, limit=args.limit)

    payload: Dict[str, Any] = {"havoc": havoc, "rtp": rtp, "limit": args.limit}
    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    out_fig = Path(args.out_fig)
    _plot_radar(havoc, rtp, out_fig)
    print(f"Saved {out_json}")
    print(f"Saved {out_fig}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
