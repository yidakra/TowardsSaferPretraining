#!/bin/bash
#SBATCH --job-name=ttp_eval
#SBATCH --partition=thin
#SBATCH --time=02:00:00
#SBATCH --mem=16G
#SBATCH --output=$SLURM_SUBMIT_DIR/logs/%x_%j.out
#SBATCH --error=$SLURM_SUBMIT_DIR/logs/%x_%j.err

# Enable strict error handling
set -euo pipefail

# Ensure logs directory exists
mkdir -p "$SLURM_SUBMIT_DIR/logs"

# Create logs directory
mkdir -p logs

module purge
module load 2023
module load Python/3.11.3-GCCcore-12.3.0

# Set project directory from environment or fallback
PROJECT_DIR="${PROJECT_DIR:-${0%/*}/..}"
PROJECT_DIR="${PROJECT_DIR:-$HOME/TowardsSaferPretraining}"

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
