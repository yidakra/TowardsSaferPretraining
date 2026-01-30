#!/bin/bash
#SBATCH --job-name=baselines_api
#SBATCH --partition=rome
#SBATCH --time=08:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/baselines_api_%j.out
#SBATCH --error=logs/baselines_api_%j.err

set -euo pipefail

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

# Change to project directory
cd "$HOME/TowardsSaferPretraining" || {
    echo "Error: Failed to change to project directory" >&2
    exit 1
}

# Activate virtual environment with error checking
source venv/bin/activate || {
    echo "Error: Failed to activate virtual environment" >&2
    exit 1
}

# Load API keys from .env if present, otherwise from example.env.
if [ -f ".env" ]; then
  set -a
  # shellcheck disable=SC1091
  source ".env"
  set +a
elif [ -f "example.env" ]; then
  set -a
  # shellcheck disable=SC1091
  source "example.env"
  set +a
fi

# Create output directories
mkdir -p results/moderation
mkdir -p results/codecarbon

# Optional CodeCarbon tracking
export CODECARBON_OUTPUT_DIR="${CODECARBON_OUTPUT_DIR:-$HOME/TowardsSaferPretraining/results/codecarbon}"
export CODECARBON_EXPERIMENT_ID="${CODECARBON_EXPERIMENT_ID:-${SLURM_JOB_ID:-}}"

# Ensure required keys exist
if [ -z "${OPENAI_API_KEY:-}" ]; then
  echo "Error: OPENAI_API_KEY is required for TTP baseline" >&2
  exit 1
fi
if [ -z "${PERSPECTIVE_API_KEY:-}" ]; then
  echo "Error: PERSPECTIVE_API_KEY is required for Perspective baseline" >&2
  exit 1
fi

# Run API baselines (Table 7 API rows): Perspective + TTP on full dataset
if python scripts/evaluate_openai_moderation.py \
  --baselines perspective ttp \
  --device cpu \
  --openai-key "$OPENAI_API_KEY" \
  --perspective-key "$PERSPECTIVE_API_KEY" \
  --output results/moderation/table7_api_results.json; then
    echo "Baselines (API) complete!"
    echo "Results saved to: results/moderation/table7_api_results.json"
else
    echo "Error: Baselines (API) failed" >&2
    exit 1
fi
