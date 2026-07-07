#!/bin/bash
#SBATCH --job-name=ttp_noise_floor
#SBATCH --partition=rome
#SBATCH --time=06:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/ttp_noise_floor_%j.out
#SBATCH --error=logs/ttp_noise_floor_%j.err

# Same-snapshot F1 noise floor for the TTP endpoint (drift experiment control).
# Runs the SAME endpoint over TTP-Eval several times back-to-back so the
# April->May swing can be shown to exceed within-snapshot decoding noise, and
# tallies system_fingerprint to confirm the backend snapshot stayed constant.
#
# Config via env (all optional):
#   NOISE_SETUP   openrouter_ttp | openai_ttp   (default openrouter_ttp)
#   NOISE_REPEATS number of identical passes    (default 3)
#   NOISE_SEED    decoding seed                  (default 12345)
#   OPENROUTER_MODEL / OPENAI_MODEL              (default openai/gpt-4o / gpt-4o)

set -euo pipefail

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

PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$HOME/TowardsSaferPretraining}}"

cd "$PROJECT_DIR" || {
  echo "Error: Failed to change to project directory: $PROJECT_DIR" >&2
  exit 1
}

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

mkdir -p results/ttp_eval_noise_floor
mkdir -p results/codecarbon

export CODECARBON_OUTPUT_DIR="${CODECARBON_OUTPUT_DIR:-$PROJECT_DIR/results/codecarbon}"
export CODECARBON_EXPERIMENT_ID="${CODECARBON_EXPERIMENT_ID:-${SLURM_JOB_ID:-}}"

NOISE_SETUP="${NOISE_SETUP:-openrouter_ttp}"
NOISE_REPEATS="${NOISE_REPEATS:-3}"
NOISE_SEED="${NOISE_SEED:-12345}"

if [ "$NOISE_SETUP" = "openrouter_ttp" ] && [ -z "${OPENROUTER_API_KEY:-}" ]; then
  echo "Error: OPENROUTER_API_KEY is required for the openrouter_ttp setup" >&2
  exit 1
fi
if [ "$NOISE_SETUP" = "openai_ttp" ] && [ -z "${OPENAI_API_KEY:-}" ]; then
  echo "Error: OPENAI_API_KEY is required for the openai_ttp setup" >&2
  exit 1
fi

# shellcheck disable=SC1091
source "$PROJECT_DIR/jobs/_wandb_args.sh"
build_wandb_args ttp-eval noise-floor drift

if python scripts/ttp_noise_floor.py \
  --setup "$NOISE_SETUP" \
  --repeats "$NOISE_REPEATS" \
  --seed "$NOISE_SEED" \
  --openrouter-model "${OPENROUTER_MODEL:-openai/gpt-4o}" \
  --openai-model "${OPENAI_MODEL:-gpt-4o}" \
  --output "results/ttp_eval_noise_floor/${NOISE_SETUP}_x${NOISE_REPEATS}.json" \
  "${WANDB_ARGS[@]}"; then
    echo "Noise-floor run complete!"
    echo "Results saved to: results/ttp_eval_noise_floor/${NOISE_SETUP}_x${NOISE_REPEATS}.json"
else
    echo "Error: Noise-floor run failed" >&2
    exit 1
fi
