#!/bin/bash
#SBATCH --job-name=ttp_eval
#SBATCH --partition=thin
#SBATCH --time=02:00:00
#SBATCH --mem=16G
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

# Enable strict error handling
set -euo pipefail

# Ensure logs directory exists
mkdir -p "$SLURM_SUBMIT_DIR/logs"

# Move SLURM output files to logs directory
if [ -f "${SLURM_JOB_NAME}_${SLURM_JOB_ID}.out" ]; then
  mv "${SLURM_JOB_NAME}_${SLURM_JOB_ID}.out" "$SLURM_SUBMIT_DIR/logs/"
fi
if [ -f "${SLURM_JOB_NAME}_${SLURM_JOB_ID}.err" ]; then
  mv "${SLURM_JOB_NAME}_${SLURM_JOB_ID}.err" "$SLURM_SUBMIT_DIR/logs/"
fi

module purge
module load 2023
module load Python/3.11.3-GCCcore-12.3.0

# Set project directory from environment or fallback
if [ -z "${PROJECT_DIR:-}" ]; then
  candidate="${0%/*}/.."
  if [ -d "$candidate" ]; then
    PROJECT_DIR="$candidate"
  else
    PROJECT_DIR="$HOME/TowardsSaferPretraining"
  fi
fi

# Change to project directory with error checking
cd "$PROJECT_DIR" || {
    echo "Error: Failed to change to project directory: $PROJECT_DIR" >&2
    exit 1
}

# Activate virtual environment with error checking
source venv/bin/activate || {
    echo "Error: Failed to activate virtual environment" >&2
    exit 1
}

# Create output directory
mkdir -p results/ttp_eval

# Run TTP evaluation on full TTP-Eval dataset
if python scripts/evaluate_ttp.py \
  --data-path data/TTP-Eval/TTPEval.tsv \
  --model gpt-4o \
  --output results/ttp_eval/ttp_results.json \
  --dimension toxic; then
    echo "TTP Evaluation Complete!"
    echo "Results saved to: results/ttp_eval/ttp_results.json"
else
    echo "Error: TTP evaluation failed" >&2
    exit 1
fi
