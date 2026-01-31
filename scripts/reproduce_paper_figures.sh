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

# Load optional repo-local environment (tokens, API keys, etc.).
# Users often store HF_TOKEN/HUGGINGFACE_HUB_TOKEN in .env.
if [ -f ".env" ]; then
  set -a
  # shellcheck disable=SC1091
  . ".env"
  set +a
fi

PY=${PYTHON:-"$REPO_ROOT/venv/bin/python"}
if [ ! -x "$PY" ]; then
  echo "Python not found at $PY. Activate venv or set PYTHON=/path/to/python." >&2
  exit 1
fi

# Llama Guard 3 is an 8B model; running it on CPU is typically impractical.
# We use FIGURE_DEVICE to decide whether to attempt the English Llama Guard baseline.

# Device selection for GPU-capable steps.
FIGURE_DEVICE=${FIGURE_DEVICE:-}
if [ -z "$FIGURE_DEVICE" ]; then
  if command -v nvidia-smi >/dev/null 2>&1; then
    FIGURE_DEVICE=cuda
  else
    FIGURE_DEVICE=cpu
  fi
fi

mkdir -p results/havoc results/rtp results/ttp_eval_multilingual results/harmformer results/ttp_eval_baselines

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

# RTP results: prefer an existing full-run JSON if present; otherwise compute one.
# - Primary: continuation-only HarmFormer judge, produced by scripts/evaluate_rtp_continuations.py
# - Fallbacks: precomputed RTP JSONs that ship with some repos/runs
if [ -z "${RTP_JSON:-}" ]; then
  RTP_CANDIDATES=(
    "results/rtp/rtp_continuations_harmformer_full.json"
    "results/rtp/google_gemma-2-9b_results.json"
    "results/rtp/meta-llama_Llama-3_2-3B_results.json"
    "results/rtp/mistralai_Mistral-7B-v0_3_results.json"
  )
  for c in "${RTP_CANDIDATES[@]}"; do
    if [ -f "$c" ]; then
      RTP_JSON="$c"
      break
    fi
  done
  RTP_JSON=${RTP_JSON:-"results/rtp/rtp_continuations_harmformer_full.json"}
fi
RTP_LIMIT=${RTP_LIMIT:-""}          # empty => full dataset
RTP_STREAMING=${RTP_STREAMING:-"1"} # 1 => avoid full dataset download
RTP_BATCH_SIZE=${RTP_BATCH_SIZE:-""}
RTP_JUDGE_TEXT=${RTP_JUDGE_TEXT:-"continuation"}
RTP_MAX_CHARS=${RTP_MAX_CHARS:-""}

# Treat tiny/stub RTP JSONs as missing (older smoke files sometimes contain only leakage_percentages).
rtp_ok=0
if [ -f "$RTP_JSON" ]; then
  rtp_ok=$("$PY" - "$RTP_JSON" "${RTP_LIMIT:-}" <<'PY'
import json
import sys

path = sys.argv[1]
limit = sys.argv[2].strip() if len(sys.argv) > 2 else ""
try:
    obj = json.load(open(path, 'r', encoding='utf-8'))
except Exception:
    print(0)
    raise SystemExit(0)

ev = obj.get('evaluation')
if not isinstance(ev, dict):
    print(0)
    raise SystemExit(0)

lp = ev.get('leakage_percentages')
ts = ev.get('total_samples')

# If RTP_LIMIT is set (smoke runs), accept any non-empty total_samples.
if limit:
  ok = isinstance(lp, dict) and isinstance(ts, int) and ts > 0
else:
  ok = isinstance(lp, dict) and isinstance(ts, int) and ts >= 1000

if ok:
    print(1)
else:
    print(0)
PY
  )
fi

if [ "$rtp_ok" != "1" ]; then
  echo "[paper-figures] Computing RTP leakage JSON at $RTP_JSON (missing/invalid)"
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

  if [ -n "$RTP_MAX_CHARS" ]; then
    cmd+=(--max-chars "$RTP_MAX_CHARS")
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

# English reference bars (best-effort):
# - HarmFormer: computed on English TTP-Eval (used as reference in multilingual plot)
# - Llama Guard: computed on English TTP-Eval (requires HF access token)
if [ ! -f "results/harmformer/harmformer_results.json" ]; then
  echo "[paper-figures] Computing HarmFormer English baseline at results/harmformer/harmformer_results.json"
  "$PY" scripts/evaluate_ttp_eval.py \
    --setups harmformer \
    --device "$FIGURE_DEVICE" \
    --dimension toxic \
    --invalid-policy exclude \
    --output results/harmformer/harmformer_results.json
fi

if [ ! -f "results/ttp_eval_baselines/llama_guard_en.json" ]; then
  if [ "$FIGURE_DEVICE" = "cpu" ] && [ "${ALLOW_LLAMA_GUARD_CPU:-0}" != "1" ]; then
    echo "[paper-figures] Skipping Llama Guard English baseline (FIGURE_DEVICE=cpu; set ALLOW_LLAMA_GUARD_CPU=1 to force, but CPU runs are typically impractical for LG3-8B)." >&2
  elif [ -n "${HF_TOKEN:-}" ] || [ -n "${HUGGINGFACE_HUB_TOKEN:-}" ]; then
    echo "[paper-figures] Computing Llama Guard English baseline at results/ttp_eval_baselines/llama_guard_en.json"
    "$PY" scripts/evaluate_ttp_eval.py \
      --setups llama_guard \
      --device "$FIGURE_DEVICE" \
      --dimension toxic \
      --invalid-policy exclude \
      --output results/ttp_eval_baselines/llama_guard_en.json
  else
    echo "[paper-figures] Skipping Llama Guard English baseline (set HF_TOKEN or HUGGINGFACE_HUB_TOKEN to enable)." >&2
  fi
fi

echo "[paper-figures] 3/3: multilingual-f1.png"
"$PY" scripts/plot_multilingual_f1.py \
  --multilingual-results-dir results/ttp_eval_multilingual \
  --llama-guard-en-json results/ttp_eval_baselines/llama_guard_en.json \
  --out multilingual-f1.png

echo "[paper-figures] Done. Generated:"
ls -1 havoc-topical-counts.png havoc-rtp-compare.png multilingual-f1.png 2>/dev/null || true
