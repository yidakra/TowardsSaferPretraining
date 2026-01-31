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

# Set project directory.
PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$HOME/TowardsSaferPretraining}}"

# Change to project directory
cd "$PROJECT_DIR" || {
  echo "Error: Failed to change to project directory: $PROJECT_DIR" >&2
  exit 1
}

# Activate virtual environment with error checking
source venv/bin/activate || {
    echo "Error: Failed to activate virtual environment" >&2
    exit 1
}

# Load API keys from .env (absolute path for Slurm jobs)
if [ -f "$HOME/TowardsSaferPretraining/.env" ]; then
  set -a
  # shellcheck disable=SC1091
  source "$HOME/TowardsSaferPretraining/.env"
  set +a
elif [ -f "$HOME/TowardsSaferPretraining/example.env" ]; then
  set -a
  # shellcheck disable=SC1091
  source "$HOME/TowardsSaferPretraining/example.env"
  set +a
fi

# Create output directories
mkdir -p results/moderation
mkdir -p results/codecarbon

# Optional CodeCarbon tracking
export CODECARBON_OUTPUT_DIR="${CODECARBON_OUTPUT_DIR:-$PROJECT_DIR/results/codecarbon}"
export CODECARBON_EXPERIMENT_ID="${CODECARBON_EXPERIMENT_ID:-${SLURM_JOB_ID:-}}"

# Ensure required keys exist
if [ -z "${OPENROUTER_API_KEY:-}" ]; then
  echo "Error: OPENROUTER_API_KEY is required for OpenRouter TTP baseline" >&2
  exit 1
fi
if [ -z "${PERSPECTIVE_API_KEY:-}" ]; then
  echo "Error: PERSPECTIVE_API_KEY is required for Perspective baseline" >&2
  exit 1
fi

# Run API baselines (Table 7 API rows): Perspective + TTP (OpenRouter) on full dataset
if python scripts/evaluate_openai_moderation.py \
  --baselines perspective ttp_openrouter \
  --device cpu \
  --openrouter-key "$OPENROUTER_API_KEY" \
  --openrouter-model "${OPENROUTER_MODEL:-openai/gpt-4o}" \
  --perspective-key "$PERSPECTIVE_API_KEY" \
  --output results/moderation/table7_api_results.json; then
    echo "Baselines (API) complete!"
    echo "Results saved to: results/moderation/table7_api_results.json"
else
    echo "Error: Baselines (API) failed" >&2
    exit 1
fi
