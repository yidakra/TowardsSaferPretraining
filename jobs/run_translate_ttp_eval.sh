#!/bin/bash
#SBATCH --job-name=translate_ttp_eval
#SBATCH --partition=gpu_a100
#SBATCH --gpus-per-node=1
#SBATCH --time=04:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/translate_ttp_eval_%j.out
#SBATCH --error=logs/translate_ttp_eval_%j.err

set -euo pipefail

mkdir -p logs

module purge
module load 2023
module load Python/3.11.3-GCCcore-12.3.0
module load CUDA/12.1.1

PROJECT_DIR="${PROJECT_DIR:-${1:-$HOME/TowardsSaferPretraining}}"
cd "$PROJECT_DIR"

source venv/bin/activate

# Optional CodeCarbon tracking
mkdir -p results/codecarbon
export CODECARBON_OUTPUT_DIR="${CODECARBON_OUTPUT_DIR:-$PROJECT_DIR/results/codecarbon}"
export CODECARBON_EXPERIMENT_ID="${CODECARBON_EXPERIMENT_ID:-${SLURM_JOB_ID:-}}"

# Translation is expensive. Defaults to the smaller NLLB model.
MODEL_ID="${MODEL_ID:-facebook/nllb-200-distilled-600M}"
SRC_LANG="${SRC_LANG:-eng_Latn}"
TGT_LANGS="${TGT_LANGS:-spa_Latn fra_Latn deu_Latn arb_Arab hin_Deva zho_Hans}"
BATCH_SIZE="${BATCH_SIZE:-8}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"

python scripts/translate_ttp_eval.py \
  --input data/TTP-Eval/TTPEval.tsv \
  --model-id "$MODEL_ID" \
  --src-lang "$SRC_LANG" \
  --tgt-langs $TGT_LANGS \
  --device cuda \
  --batch-size "$BATCH_SIZE" \
  --max-new-tokens "$MAX_NEW_TOKENS"
