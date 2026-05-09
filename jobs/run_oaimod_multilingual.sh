#!/bin/bash
#SBATCH --job-name=oaimod_multilingual
#SBATCH --partition=gpu_a100
#SBATCH --gpus-per-node=1
#SBATCH --time=04:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/oaimod_multilingual_%j.out
#SBATCH --error=logs/oaimod_multilingual_%j.err

set -euo pipefail

mkdir -p logs

module purge
module load 2023
module load Python/3.11.3-GCCcore-12.3.0
module load CUDA/12.1.1

PROJECT_DIR="${PROJECT_DIR:-${1:-$HOME/TowardsSaferPretraining}}"
cd "$PROJECT_DIR"

source venv/bin/activate
set -a; source "$HOME/TowardsSaferPretraining/.env" 2>/dev/null || true; set +a

mkdir -p results/codecarbon
export CODECARBON_OUTPUT_DIR="${CODECARBON_OUTPUT_DIR:-$PROJECT_DIR/results/codecarbon}"
export CODECARBON_EXPERIMENT_ID="${CODECARBON_EXPERIMENT_ID:-${SLURM_JOB_ID:-}}"

LANGS="${LANGS:-spa_Latn fra_Latn deu_Latn arb_Arab hin_Deva zho_Hans}"
TRANSLATED_DIR="${TRANSLATED_DIR:-data/moderation-api-release/translated/nllb-200-3.3B}"
EVAL_OUTPUT_DIR="${EVAL_OUTPUT_DIR:-results/moderation_multilingual}"

mkdir -p "$TRANSLATED_DIR" "$EVAL_OUTPUT_DIR"

echo "=== Phase 1: translate OAI Moderation into $LANGS ==="
python scripts/translate_openai_moderation.py \
    --tgt-langs $LANGS \
    --output-dir "$TRANSLATED_DIR" \
    --device cuda \
    --batch-size 16

echo "=== Phase 2: evaluate HarmFormer + Llama Guard on translated samples ==="
for lang in $LANGS; do
    in_path="$TRANSLATED_DIR/samples-1680_${lang}.jsonl.gz"
    if [[ ! -f "$in_path" ]]; then
        echo "Missing translated file: $in_path; aborting"
        exit 1
    fi
    for setup in harmformer llama_guard; do
        out="$EVAL_OUTPUT_DIR/${setup}_${lang}.json"
        echo "--- $setup $lang ---"
        python scripts/evaluate_openai_moderation.py \
            --data-path "$in_path" \
            --baselines "$setup" \
            --device cuda \
            --output "$out"
    done
done

echo "=== Done. Wrote eval results to $EVAL_OUTPUT_DIR ==="
