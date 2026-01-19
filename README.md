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
├── results/              # outputs (JSON) + CodeCarbon emissions
├── logs/                 # Slurm stdout/stderr (when using jobs/*.sh)
├── LICENSE
└── README.md
```

- **data/**: Contains the datasets used in the paper.
- **prompts/**: Includes prompts used for evaluating TTP and HAVOC.
- **scripts/**: Local, direct entrypoints for evaluation and report generation.
- **jobs/**: Snellius Slurm wrappers for GPU-backed runs.
- **LICENSE**: License information for the datasets and code.

## Getting Started

1. Clone the repository:
    ```bash
    git clone https://github.com/themendu/TowardsSaferPretraining.git
    cd TowardsSaferPretraining
    ```
2. Review the `data/README.md` for dataset details and usage instructions.
3. Use the prompts folder in case you need to evaluate TTP. The prompts are styled in OpenAI ChatML format.
4. To access HarmFormer’s predictions on the entire C4 dataset, please visit [this link](https://huggingface.co/datasets/themendu/SafeC4).
5. We release HarmFormer [here in HuggingFace](https://huggingface.co/themendu/HarmFormer)

## Reproducing paper results (what we currently support)

### Common setup (local or Snellius)

```bash
python -m venv venv
source venv/bin/activate
python -m pip install -U pip
python -m pip install -r requirements.txt
```

- **API keys via env**:
  - **OpenAI TTP**: `OPENAI_API_KEY` (Table 3, Table 4 “TTP” row, HAVOC judge=TTP)
  - **Gemini TTP**: `GEMINI_API_KEY` (+ optional `GEMINI_MODEL`, default `gemini-2.0-flash`) (Table 4 Gemini row)
  - **Perspective**: `PERSPECTIVE_API_KEY` (Table 4 Perspective row)
    - Or set `ENABLE_PERSPECTIVE_WITH_GEMINI_KEY=1` to reuse `GEMINI_API_KEY` for Perspective if that key/project is enabled for the Perspective API.
  - **Llama Guard (HF gated model)**: `HUGGINGFACE_HUB_TOKEN` (or `HF_TOKEN`) (baseline comparisons)
- **Emissions tracking (CodeCarbon)**: enabled by default. Outputs go to `results/codecarbon/`. Set `DISABLE_CODECARBON=1` to disable.

### Local (CPU/API-only)

#### Table 4 (Toxic dimension): Perspective vs TTP variants on TTP-Eval

- **Perspective + Gemini TTP (no GPU required)**:

```bash
python scripts/evaluate_ttp_eval.py \
  --setups perspective gemini_ttp \
  --perspective-key "$PERSPECTIVE_API_KEY" \
  --gemini-key "$GEMINI_API_KEY" \
  --gemini-model "${GEMINI_MODEL:-gemini-2.0-flash}" \
  --dimension toxic \
  --output results/ttp_eval_baselines/results.json
```

- **Perspective + OpenAI TTP (no GPU required; requires OpenAI quota/billing)**:

```bash
python scripts/evaluate_ttp_eval.py \
  --setups perspective openai_ttp \
  --perspective-key "$PERSPECTIVE_API_KEY" \
  --openai-key "$OPENAI_API_KEY" \
  --dimension toxic \
  --output results/ttp_eval_baselines/results.json
```

#### Table 3 (Toxic dimension): TTP quality on TTP-Eval (OpenAI)

```bash
python scripts/evaluate_ttp_eval.py \
  --data-path data/TTP-Eval/TTPEval.tsv \
  --setups openai_ttp \
  --openai-model gpt-4o \
  --dimension toxic \
  --output results/ttp_eval/results.json
```

#### Table 10: HAVOC leakage from released `havoc_modeleval.tsv` (no model downloads)

Run one model column:

```bash
python scripts/evaluate_havoc_modeleval.py \
  --data-path data/HAVOC/havoc.tsv \
  --modeleval-path data/HAVOC/havoc_modeleval.tsv \
  --model-key gemma_2b \
  --output results/havoc/gemma_2b_results.json
```

Or run all model keys locally:

```bash
for k in gemma_2b gemma_9b gemma_27b llama_1b llama_3b mistral_7b; do \
  python scripts/evaluate_havoc_modeleval.py --model-key \"$k\" --output \"results/havoc/${k}_results.json\"; \
done
```

#### Reproduction report (prints to stdout)

```bash
python scripts/generate_report.py
```

### Snellius (Slurm)

Only needed for **GPU-backed runs** on Snellius (Slurm). If you’re running CPU/API locally, use the “Local” section above.

#### Setup venv on Snellius (once)

```bash
sbatch setup_env.sh
```

#### Table 6: HarmFormer quality on TTP-Eval (GPU recommended)

```bash
sbatch jobs/run_harmformer_eval.sh
```

#### Table 3 (Toxic dimension): TTP quality on TTP-Eval (OpenAI)

```bash
sbatch jobs/run_ttp_eval.sh
```

#### Baseline comparison job (GPU; runs what credentials enable)

This job can cover local baselines (HarmFormer, Llama Guard prompt variants) without API keys:

```bash
sbatch jobs/run_baselines_local.sh
```

If you want the API rows (Perspective + OpenAI TTP), run:

```bash
sbatch jobs/run_baselines_api.sh
```

#### Table 4 (Toxic dimension): Perspective vs OpenAI TTP (API rows on TTP-Eval)

```bash
sbatch jobs/run_ttp_eval_api.sh
```

#### Table 7 (OpenAI Moderation test set): Perspective/Llama Guard/TTP/HarmFormer

First, fetch the dataset (one-time):

```bash
git clone --depth 1 https://github.com/openai/moderation-api-release.git data/moderation-api-release
```

Then run:

```bash
python scripts/evaluate_openai_moderation.py \
  --output results/moderation/table7_results.json
```

#### Table 4 local-model rows (Gemma 2 27B; optional R1 model)

Run Table 4 with local Transformers models on Snellius (GPU). This is needed to reproduce the paper’s non-API rows
like **Gemma 2 27B**. The paper’s “R1 - LLaMa 32B” model id is not stable across releases; provide it via `R1_MODEL_ID`.

```bash
sbatch jobs/run_ttp_eval_local.sh
```

Environment variables:
- `GEMMA_2_27B_MODEL_ID` (default: `google/gemma-2-27b-it`)
- `R1_MODEL_ID` (optional; set to the HF model id you intend to match)
- `TTP_LOCAL_QUANTIZATION` (optional: `none|8bit|4bit`; requires `bitsandbytes` if not `none`)

#### Table 7 Slurm wrapper (GPU)

```bash
sbatch jobs/run_table7_moderation.sh
```

## License

See the [LICENSE](./LICENSE) file for details.

## Contact

For questions or contributions, please open an issue or contact the maintainers.
