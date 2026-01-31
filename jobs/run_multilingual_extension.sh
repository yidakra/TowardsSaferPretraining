#!/bin/bash
#SBATCH --job-name=multilingual_extension
#SBATCH --partition=gpu_a100
#SBATCH --gpus-per-node=1
#SBATCH --time=01:30:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/multilingual_extension_%j.out
#SBATCH --error=logs/multilingual_extension_%j.err

set -euo pipefail

mkdir -p logs

module purge
module load 2023
module load Python/3.11.3-GCCcore-12.3.0
module load CUDA/12.1.1

PROJECT_DIR="${PROJECT_DIR:-${1:-$HOME/TowardsSaferPretraining}}"
cd "$PROJECT_DIR"


source venv/bin/activate
# Load environment variables from .env file (absolute path for Slurm)
set -a; source "$HOME/TowardsSaferPretraining/.env" 2>/dev/null || true; set +a

# Optional CodeCarbon tracking
mkdir -p results/codecarbon
export CODECARBON_OUTPUT_DIR="${CODECARBON_OUTPUT_DIR:-$PROJECT_DIR/results/codecarbon}"
export CODECARBON_EXPERIMENT_ID="${CODECARBON_EXPERIMENT_ID:-${SLURM_JOB_ID:-}}"

# Choose which translated directory to evaluate.
TRANSLATED_DIR="${TRANSLATED_DIR:-data/TTP-Eval/translated/nllb-200-3.3B}"
LANGS="${LANGS:-spa_Latn fra_Latn deu_Latn arb_Arab hin_Deva zho_Hans}"
SETUPS="${SETUPS:-harmformer llama_guard}"
LIMIT="${LIMIT:-}"

# If you include `llama_guard` in SETUPS, you need access to the gated model.
# Export one of: HF_TOKEN / HUGGINGFACE_HUB_TOKEN.

CMD=(python scripts/evaluate_ttp_eval_multilingual.py \
  --translated-dir "$TRANSLATED_DIR" \
  --langs $LANGS \
  --setups $SETUPS \
  --device cuda \
  --dimension toxic \
  --output-dir results/ttp_eval_multilingual)

if [[ -n "$LIMIT" ]]; then
  CMD+=(--limit "$LIMIT")
fi

"${CMD[@]}"
