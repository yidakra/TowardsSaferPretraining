#!/usr/bin/env bash
set -euo pipefail

# Generates paper figures (PNG) from locally generated JSON results.
# Note: PNG/SVG files are gitignored by design.

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$REPO_ROOT"

PY=${PYTHON:-"$REPO_ROOT/venv/bin/python"}

if [ ! -x "$PY" ]; then
  echo "Python not found at $PY. Activate venv or set PYTHON=/path/to/python." >&2
  exit 1
fi


echo "[figures] HAVOC topical counts (from data/HAVOC/havoc.tsv)"
$PY scripts/plot_havoc_topical_counts.py

echo "[figures] HAVOC vs RTP compare (requires results/havoc/*_results.json + one RTP JSON)"
if compgen -G "results/havoc/*_results.json" >/dev/null && [ -f "results/rtp/meta-llama_Llama-3_2-3B_results.json" ]; then
  $PY scripts/plot_havoc_rtp_compare.py
else
  echo "  - Skipping havoc-rtp-compare.png (missing inputs under results/)." >&2
fi

echo "[figures] Multilingual F1 (requires results/ttp_eval_multilingual/*.json)"
if compgen -G "results/ttp_eval_multilingual/harmformer_*.json" >/dev/null && compgen -G "results/ttp_eval_multilingual/llama_guard_*.json" >/dev/null; then
  $PY scripts/plot_multilingual_f1.py
else
  echo "  - Skipping multilingual-f1.png (missing multilingual result JSONs under results/)." >&2
fi

# Optional appendix figure
if [ -f "results/rtp/havoc_rtp_harms.json" ]; then
  $PY scripts/plot_rtp_havoc_harms_radar.py || true
fi

echo "Generated figures:"
ls -1 havoc-topical-counts.png havoc-rtp-compare.png multilingual-f1.png 2>/dev/null || true
