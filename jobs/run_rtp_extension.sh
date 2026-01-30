#!/bin/bash
#SBATCH --job-name=rtp_extension
#SBATCH --partition=gpu_a100
#SBATCH --gpus-per-node=1
#SBATCH --time=01:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/rtp_extension_%j.out
#SBATCH --error=logs/rtp_extension_%j.err

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

LIMIT="${LIMIT:-}"

CMD=(python scripts/evaluate_rtp_continuations.py \
  --device cuda \
  --batch-size 32 \
  --output results/rtp/rtp_continuations_harmformer.json)

if [[ -n "$LIMIT" ]]; then
  CMD+=(--limit "$LIMIT")
fi

"${CMD[@]}"
