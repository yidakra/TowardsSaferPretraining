#!/bin/bash
set -euo pipefail

#SBATCH --job-name=baselines
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=1
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/baselines_%j.out
#SBATCH --error=logs/baselines_%j.err

# Create logs directory
mkdir -p logs

module purge
module load 2023 || {
    echo "Error: Failed to load module 2023" >&2
    exit 1
}
module load Python/3.11.3-GCCcore-12.3.0 || {
    echo "Error: Failed to load module Python/3.11.3-GCCcore-12.3.0" >&2
    exit 1
}
module load CUDA/12.1.1 || {
    echo "Error: Failed to load module CUDA/12.1.1" >&2
    exit 1
}

# Change to project directory
cd "$HOME/TowardsSaferPretraining" || {
    echo "Error: Failed to change to project directory" >&2
    exit 1
}

# Activate virtual environment
source venv/bin/activate || {
    echo "Error: Failed to activate virtual environment" >&2
    exit 1
}

# Create output directory
mkdir -p results/baselines

# Run baseline comparison
# This reproduces Table 4 (TTP vs Perspective API) and Table 7 (OpenAI Moderation dataset)
if python scripts/compare_baselines.py \
  --output results/baselines/baseline_comparison.json; then
    echo "Baseline comparison complete!"
    echo "Results saved to: results/baselines/baseline_comparison.json"
else
    echo "Error: Baseline comparison failed" >&2
    exit 1
fi
