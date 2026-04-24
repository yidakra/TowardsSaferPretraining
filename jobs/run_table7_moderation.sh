#!/bin/bash
#SBATCH --job-name=table7_moderation
#SBATCH --partition=gpu_a100
#SBATCH --gpus-per-node=1
#SBATCH --time=04:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

mkdir -p logs

module purge
module load 2023
module load Python/3.11.3-GCCcore-12.3.0
module load CUDA/12.1.1

PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$HOME/TowardsSaferPretraining}}"
cd "$PROJECT_DIR"
source venv/bin/activate

# Load env keys if present
if [ -f ".env" ]; then
  set -a
  # shellcheck disable=SC1091
  source ".env"
  set +a
fi

mkdir -p results/moderation results/codecarbon
export CODECARBON_OUTPUT_DIR="${CODECARBON_OUTPUT_DIR:-$PROJECT_DIR/results/codecarbon}"
export CODECARBON_EXPERIMENT_ID="${CODECARBON_EXPERIMENT_ID:-${SLURM_JOB_ID:-}}"

WANDB_ARGS=()
if [ "${WANDB_ENABLED:-0}" = "1" ]; then
  WANDB_ARGS+=(--wandb --wandb-project "${WANDB_PROJECT:-TowardsSaferPretraining}")
  if [ -n "${WANDB_ENTITY:-}" ]; then WANDB_ARGS+=(--wandb-entity "$WANDB_ENTITY"); fi
  if [ -n "${WANDB_GROUP:-}" ]; then WANDB_ARGS+=(--wandb-group "$WANDB_GROUP"); fi
  if [ -n "${WANDB_MODE:-}" ]; then WANDB_ARGS+=(--wandb-mode "$WANDB_MODE"); fi
  WANDB_ARGS+=(--wandb-tags moderation table7 slurm)
fi

python scripts/evaluate_openai_moderation.py \
  --output "results/moderation/table7_results.json" \
  --device cuda \
  "${WANDB_ARGS[@]}"
