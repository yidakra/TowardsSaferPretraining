## Overview

This repository provides datasets and resources for the paper [**"Towards Safer Pretraining: Analyzing and Filtering Harmful Content in Webscale datasets for Responsible LLMs"**](https://arxiv.org/pdf/2505.02009). The datasets are curated to support research on safer and more responsible pretraining of language models.

## Folder Structure

```
TowardsSaferPretraining/
├── data/
│   ├── README.md
│   ├── HAVOC/
│   └── TTP-Eval/
├── jobs/                 # Snellius (Slurm) job entrypoints
├── prompts/
│   ├── HAVOC/
│   └── TTP/
├── scripts/              # local entrypoints (CPU/API/GPU depending on script)
├── src/                  # library code (clients, benchmarks, loaders, utils)
├── results/              # generated outputs (JSON) + CodeCarbon emissions (not versioned)
├── LICENSE
└── README.md
```

- **data/**: Contains the datasets used in the paper.
- **prompts/**: Includes prompts used for evaluating TTP and HAVOC.
- **scripts/**: Local, direct entrypoints for evaluation and report generation.
- **jobs/**: Snellius Slurm wrappers for GPU-backed runs.
- **LICENSE**: License information for the datasets and code.

## Getting Started

1. Clone the repository.
2. Review the `data/README.md` for dataset details and usage instructions.
3. Use the prompts folder in case you need to evaluate TTP. The prompts are styled in OpenAI ChatML format.
4. To access HarmFormer’s predictions on the entire C4 dataset, please visit [this link](https://huggingface.co/datasets/themendu/SafeC4).
5. To access HarmFormer, visit [this link](https://huggingface.co/themendu/HarmFormer)

## Reproducing paper results

This README intentionally provides **one** end-to-end reproduction path.
Reviewers are expected to (re)compute **all** JSON results before generating plots/tables.

### 0) Prerequisites

- Snellius access (Slurm) with an A100 partition.
- A repo-local `.env` file (see `.env.example`) containing all required credentials.
  The job scripts and bash helpers will automatically load `.env`.

Required environment variables in `.env`:
- `OPENROUTER_API_KEY` (TTP via OpenRouter; Table 3 and Table 7)
- `PERSPECTIVE_API_KEY` (Perspective baseline; Table 7)
- `GEMINI_API_KEY` (Gemini TTP row in Table 4)
- `HUGGINGFACE_HUB_TOKEN` (or `HF_TOKEN`) (HF gated models: Llama Guard + some local baselines)

Note: OpenRouter calls require an account with sufficient credits. If you cannot fund OpenRouter,
you can still reproduce the local-only parts (e.g., HarmFormer, Llama Guard, HAVOC/RTP), but Table 3
and the TTP rows in Table 7 will be unavailable.

One-time dataset fetch (OpenAI Moderation test set used in Table 7):

```bash
git clone --depth 1 https://github.com/openai/moderation-api-release.git data/moderation-api-release
```

### 1) Environment setup

Create the Python environment on Snellius (once):

```bash
sbatch setup_env.sh
```

After the job finishes:

```bash
source venv/bin/activate
set -a; source .env; set +a
pytest -q
```

### 2) Compute all result JSONs (Tables)

Submit the following Slurm jobs and wait for them to complete:

```bash
mkdir -p logs

# Table 3: TTP quality on TTP-Eval (OpenRouter)
sbatch jobs/run_ttp_eval.sh

# Table 6: HarmFormer quality on TTP-Eval (GPU)
sbatch jobs/run_harmformer_eval.sh

# Table 4: local-model baselines on TTP-Eval (2× A100)
# Requires HF access for the specified model ids.
sbatch jobs/run_ttp_eval_local.sh

# Table 7: OpenAI Moderation baselines (split into local vs API for clarity)
sbatch jobs/run_baselines_local.sh
sbatch jobs/run_baselines_api.sh

# NOTE: `jobs/run_baselines_api.sh` already includes the OpenRouter TTP baseline.
# If you want to run it standalone (CPU; uses OPENROUTER_API_KEY), use:
# python scripts/evaluate_openai_moderation.py \
#   --baselines ttp_openrouter \
#   --device cpu \
#   --output results/moderation/table7_ttp_openrouter.json

# Table 10: HAVOC leakage from released havoc_modeleval.tsv (CPU; no model inference)
for k in gemma_2b gemma_9b gemma_27b llama_1b llama_3b mistral_7b; do \
  python scripts/evaluate_havoc_modeleval.py \
    --model-key "$k" \
    --output "results/havoc/${k}_results.json"; \
done

# Table 4: API-backed TTP baselines on TTP-Eval
python scripts/evaluate_ttp_eval.py \
  --data-path data/TTP-Eval/TTPEval.tsv \
  --setups openrouter_ttp \
  --openrouter-key "$OPENROUTER_API_KEY" \
  --openrouter-model "${OPENROUTER_MODEL:-openai/gpt-4o}" \
  --dimension toxic \
  --output results/ttp_eval_baselines/results.json

python scripts/evaluate_ttp_eval.py \
  --data-path data/TTP-Eval/TTPEval.tsv \
  --setups gemini_ttp \
  --gemini-key "$GEMINI_API_KEY" \
  --gemini-model "${GEMINI_MODEL:-gemini-2.0-flash}" \
  --dimension toxic \
  --output results/ttp_eval_baselines/table4_gemini_ttp.json
```

### 3) Compute all result JSONs (Figures)

These runs produce the JSON inputs needed for the paper figures.

```bash
# Multilingual robustness JSONs (writes under results/ttp_eval_multilingual/; GPU)
sbatch jobs/run_multilingual_extension.sh

# RTP leakage JSON (writes under results/rtp/; GPU)
sbatch jobs/run_rtp_extension.sh

# English Llama Guard baseline on TTP-Eval for the multilingual plot (GPU)
sbatch --job-name=llama_guard_en \
  --partition=gpu_a100 \
  --gpus-per-node=1 \
  --time=01:00:00 \
  --mem=64G \
  --cpus-per-task=4 \
  --output=logs/llama_guard_en_%j.out \
  --error=logs/llama_guard_en_%j.err \
  --wrap "module purge && module load 2023 && module load Python/3.11.3-GCCcore-12.3.0 && module load CUDA/12.1.1 && cd $PWD && source venv/bin/activate && set -a && source .env && set +a && mkdir -p results/ttp_eval_baselines && python scripts/evaluate_ttp_eval.py --data-path data/TTP-Eval/TTPEval.tsv --setups llama_guard --device cuda --dimension toxic --invalid-policy exclude --output results/ttp_eval_baselines/llama_guard_en.json"
```

### 4) Generate plots and the reproduction report

After all jobs above have completed and JSON outputs are present under `results/`:

```bash
python scripts/generate_report.py > results/reproduction_report.txt
bash scripts/generate_figures.sh
```

Expected generated files:
- `results/reproduction_report.txt`
- `havoc-topical-counts.png`
- `havoc-rtp-compare.png`
- `multilingual-f1.png`

## License

See the [LICENSE](./LICENSE) file for details.

## Contact

For questions or contributions, please open an issue or contact the maintainers.
