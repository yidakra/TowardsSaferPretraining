#!/usr/bin/env bash
set -euo pipefail

# One-command reproduction of the three PNG figures used by draft_current.tex.
#
# Generates:
#   - havoc-topical-counts.png
#   - havoc-rtp-compare.png
#   - multilingual-f1.png
#
# This script will (re)run prerequisite evaluations if their JSON inputs are missing.
# Outputs under results/ are intentionally not versioned.

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$REPO_ROOT"

PY=${PYTHON:-"$REPO_ROOT/venv/bin/python"}
if [ ! -x "$PY" ]; then
  echo "Python not found at $PY. Activate venv or set PYTHON=/path/to/python." >&2
  exit 1
fi

# Device selection for GPU-capable steps.
FIGURE_DEVICE=${FIGURE_DEVICE:-}
if [ -z "$FIGURE_DEVICE" ]; then
  if command -v nvidia-smi >/dev/null 2>&1; then
    FIGURE_DEVICE=cuda
  else
    FIGURE_DEVICE=cpu
  fi
fi

mkdir -p results/havoc results/rtp results/ttp_eval_multilingual

# ----------------------------
# 1) HAVOC topical counts
# ----------------------------
echo "[paper-figures] 1/3: havoc-topical-counts.png"
"$PY" scripts/plot_havoc_topical_counts.py

# ----------------------------
# 2) HAVOC leakage JSONs + RTP JSON + compare plot
# ----------------------------
# HAVOC leakage JSONs are computed from released `havoc_modeleval.tsv` (no model downloads).
HAVOC_KEYS=(gemma_2b gemma_9b gemma_27b llama_1b llama_3b mistral_7b)

missing_havoc=0
for k in "${HAVOC_KEYS[@]}"; do
  if [ ! -f "results/havoc/${k}_results.json" ]; then
    missing_havoc=1
    break
  fi
done

if [ "$missing_havoc" = "1" ]; then
  echo "[paper-figures] Computing HAVOC leakage JSONs under results/havoc/"
  for k in "${HAVOC_KEYS[@]}"; do
    "$PY" scripts/evaluate_havoc_modeleval.py \
      --model-key "$k" \
      --output "results/havoc/${k}_results.json"
  done
fi

RTP_JSON=${RTP_JSON:-"results/rtp/rtp_continuations_harmformer.json"}
RTP_LIMIT=${RTP_LIMIT:-""}          # empty => full dataset
RTP_STREAMING=${RTP_STREAMING:-"1"} # 1 => avoid full dataset download
RTP_BATCH_SIZE=${RTP_BATCH_SIZE:-""}
RTP_JUDGE_TEXT=${RTP_JUDGE_TEXT:-"continuation"}

if [ ! -f "$RTP_JSON" ]; then
  echo "[paper-figures] Computing RTP leakage JSON at $RTP_JSON"
  cmd=("$PY" scripts/evaluate_rtp_continuations.py \
    --device "$FIGURE_DEVICE" \
    --judge-text "$RTP_JUDGE_TEXT" \
    --output "$RTP_JSON")

  if [ "$RTP_STREAMING" = "1" ]; then
    cmd+=(--streaming)
  fi
  if [ -n "$RTP_LIMIT" ]; then
    cmd+=(--limit "$RTP_LIMIT")
  fi

  if [ -n "$RTP_BATCH_SIZE" ]; then
    cmd+=(--batch-size "$RTP_BATCH_SIZE")
  else
    if [ "$FIGURE_DEVICE" = "cuda" ]; then
      cmd+=(--batch-size 32)
    else
      cmd+=(--batch-size 16)
    fi
  fi

  "${cmd[@]}"
fi

echo "[paper-figures] 2/3: havoc-rtp-compare.png"
"$PY" scripts/plot_havoc_rtp_compare.py \
  --havoc-results-dir results/havoc \
  --rtp-results-json "$RTP_JSON" \
  --out havoc-rtp-compare.png

# ----------------------------
# 3) Multilingual evaluation JSONs + plot
# ----------------------------
TRANSLATED_DIR=${TRANSLATED_DIR:-""}
if [ -z "$TRANSLATED_DIR" ]; then
  if [ -d "data/TTP-Eval/translated/nllb-200-3.3B" ]; then
    TRANSLATED_DIR="data/TTP-Eval/translated/nllb-200-3.3B"
  else
    TRANSLATED_DIR="data/TTP-Eval/translated/nllb-200-distilled-600M"
  fi
fi

LANGS=${LANGS:-"spa_Latn fra_Latn deu_Latn arb_Arab hin_Deva zho_Hans"}
MULTILINGUAL_SETUPS=${MULTILINGUAL_SETUPS:-"harmformer llama_guard"}
MULTILINGUAL_LIMIT=${MULTILINGUAL_LIMIT:-""}

# Heuristic: only run multilingual eval if at least one expected output is missing.
need_multi=0
for lang in $LANGS; do
  for setup in $MULTILINGUAL_SETUPS; do
    if [ ! -f "results/ttp_eval_multilingual/${setup}_${lang}.json" ]; then
      need_multi=1
      break
    fi
  done
  if [ "$need_multi" = "1" ]; then
    break
  fi
done

if [ "$need_multi" = "1" ]; then
  echo "[paper-figures] Computing multilingual TTP-Eval JSONs under results/ttp_eval_multilingual/"
  cmd=("$PY" scripts/evaluate_ttp_eval_multilingual.py \
    --translated-dir "$TRANSLATED_DIR" \
    --langs $LANGS \
    --setups $MULTILINGUAL_SETUPS \
    --device "$FIGURE_DEVICE" \
    --dimension toxic \
    --output-dir results/ttp_eval_multilingual)

  if [ -n "$MULTILINGUAL_LIMIT" ]; then
    cmd+=(--limit "$MULTILINGUAL_LIMIT")
  fi

  "${cmd[@]}"
fi

echo "[paper-figures] 3/3: multilingual-f1.png"
"$PY" scripts/plot_multilingual_f1.py \
  --multilingual-results-dir results/ttp_eval_multilingual \
  --out multilingual-f1.png

echo "[paper-figures] Done. Generated:"
ls -1 havoc-topical-counts.png havoc-rtp-compare.png multilingual-f1.png 2>/dev/null || true
