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

1. Clone the repository:
    ```bash
    git clone https://github.com/themendu/TowardsSaferPretraining.git
    cd TowardsSaferPretraining
    ```
2. Review the `data/README.md` for dataset details and usage instructions.
3. Use the prompts folder in case you need to evaluate TTP. The prompts are styled in OpenAI ChatML format.
4. To access HarmFormer’s predictions on the entire C4 dataset, please visit [this link](https://huggingface.co/datasets/themendu/SafeC4).
5. We release HarmFormer [here in HuggingFace](https://huggingface.co/themendu/HarmFormer)

## Reproducing paper results

### Common setup (local or Snellius)

```bash
python -m venv venv
source venv/bin/activate
python -m pip install -U pip
python -m pip install -r requirements.txt
```

All commands below assume you are in the same shell where you ran `source venv/bin/activate`.

If you prefer not to activate the venv, you can prefix commands with:
```bash
PY=./venv/bin/python
```
and then replace `python ...` with `$PY ...`.

- **API keys via env**:
  - **OpenAI TTP**: `OPENAI_API_KEY` (Table 3, Table 4 “TTP” row, HAVOC judge=TTP)
  - **OpenRouter TTP**: `OPENROUTER_API_KEY` (+ optional `OPENROUTER_MODEL`, default `openai/gpt-4o`) (Table 3/4 via `--setups openrouter_ttp`)
  - **Gemini TTP**: `GEMINI_API_KEY` (+ optional `GEMINI_MODEL`, default `gemini-2.0-flash`) (Table 4 Gemini row)
  - **Llama Guard (HF gated model)**: `HUGGINGFACE_HUB_TOKEN` (or `HF_TOKEN`) (baseline comparisons)
- **Emissions tracking (CodeCarbon)**: enabled by default. Outputs go to `results/codecarbon/`. Set `DISABLE_CODECARBON=1` to disable.

### Local (CPU/API-only)

#### Table 4 (Toxic dimension): (Perspective omitted)

Table 4 in the paper includes a Perspective row. We do **not** run Perspective by default, but you can still reproduce
the other Table 4 baselines on TTP-Eval:

- **TTP (OpenAI)**:

```bash
python scripts/evaluate_ttp_eval.py \
  --setups openai_ttp \
  --openai-key "$OPENAI_API_KEY" \
  --openai-model gpt-4o \
  --dimension toxic \
  --output results/ttp_eval/ttp_results.json
```

- **TTP (OpenRouter)**:

```bash
python scripts/evaluate_ttp_eval.py \
  --setups openrouter_ttp \
  --openrouter-key "$OPENROUTER_API_KEY" \
  --openrouter-model "${OPENROUTER_MODEL:-openai/gpt-4o}" \
  --dimension toxic \
  --output results/ttp_eval_baselines/results.json
```

- **HarmFormer (no API)**:

```bash
python scripts/evaluate_ttp_eval.py \
  --setups harmformer \
  --device cuda \
  --dimension toxic \
  --output results/ttp_eval_baselines/results.json
```

Optional (requires HF access to gated models):
- `--setups llama_guard`
- `--setups local_ttp --local-model google/gemma-2-27b-it`

- **Gemini TTP (Gemini 2.0 Flash)**:

```bash
python scripts/evaluate_ttp_eval.py \
  --setups gemini_ttp \
  --gemini-key "$GEMINI_API_KEY" \
  --gemini-model "gemini-2.0-flash" \
  --dimension toxic \
  --output results/ttp_eval_baselines/table4_gemini_ttp.json
```

- **Local TTP (Gemma 2 27B + LLaMa 32B)** (requires HF access + `HUGGINGFACE_HUB_TOKEN`):
  - Set `GEMMA_2_27B_MODEL_ID` (default: `google/gemma-2-27b-it`)
  - Set `LLAMA_32B_MODEL_ID` (paper “R1 32B” row) to `deepseek-ai/DeepSeek-R1-Distill-Qwen-32B` (public 32B distill release)
  - Then run: `sbatch jobs/run_ttp_eval_local.sh` (requests **2× A100** to fit these models without quantization)

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
  --model-key gemma_9b \
  --output results/havoc/gemma_9b_results.json
```

Or run all model keys locally:

```bash
for k in gemma_2b gemma_9b gemma_27b llama_1b llama_3b mistral_7b; do \
  python scripts/evaluate_havoc_modeleval.py \
    --model-key "$k" \
    --output "results/havoc/${k}_results.json"; \
done
```

#### Reproduction report (prints to stdout)

```bash
python scripts/generate_report.py
```

#### Regenerating figures (PNG; not versioned)

- `havoc-topical-counts.png`
- `havoc-rtp-compare.png`
- `multilingual-f1.png`

One-command reproduction (runs prerequisite evaluations if missing):

```bash
bash scripts/reproduce_paper_figures.sh
```

Useful environment variables:
- `FIGURE_DEVICE=cuda|cpu` (default: auto-detect)
- `RTP_LIMIT=50000` (optional; empty => full dataset)
- `RTP_STREAMING=1` (default: 1)
- `TRANSLATED_DIR=data/TTP-Eval/translated/nllb-200-3.3B` (optional)

Lightweight plotting only (requires precomputed JSON inputs under `results/`):

```bash
bash scripts/generate_figures.sh
```

### Multilingual extension

This extension evaluates toxicity classification on *translated* TTP-Eval (TTP-Eval multilingual). The repo already includes translated TSVs under `data/TTP-Eval/translated/`.

**Optional: regenerate translations (slow; GPU recommended)**

```bash
python scripts/translate_ttp_eval.py \
  --model-id facebook/nllb-200-distilled-600M \
  --src-lang eng_Latn \
  --tgt-langs spa_Latn fra_Latn deu_Latn arb_Arab hin_Deva zho_Hans
```

Expected output files (TSV):
- `data/TTP-Eval/translated/nllb-200-distilled-600M/TTPEval_spa_Latn.tsv` (and the other requested languages)

**Run multilingual evaluation (writes one JSON per language per setup)**

Required env vars (only if you run Llama Guard):
- `HF_TOKEN` or `HUGGINGFACE_HUB_TOKEN`

```bash
python scripts/evaluate_ttp_eval_multilingual.py \
  --translated-dir data/TTP-Eval/translated/nllb-200-3.3B \
  --langs spa_Latn fra_Latn deu_Latn arb_Arab hin_Deva zho_Hans \
  --setups harmformer llama_guard \
  --device cuda \
  --dimension toxic \
  --output-dir results/ttp_eval_multilingual
```

Expected output files (JSON):
- `results/ttp_eval_multilingual/harmformer_spa_Latn.json`
- `results/ttp_eval_multilingual/llama_guard_spa_Latn.json`
- (same pattern for the other languages)

To regenerate the multilingual plot:

```bash
python scripts/plot_multilingual_f1.py \
  --multilingual-results-dir results/ttp_eval_multilingual \
  --out multilingual-f1.png
```

### RTP extension

This extension evaluates “leakage” on RealToxicityPrompts (RTP) using the dataset-provided continuations (no generation) and judges toxicity with HarmFormer.

```bash
python scripts/evaluate_rtp_continuations.py \
  --device cuda \
  --batch-size 32 \
  --output results/rtp/rtp_continuations_harmformer.json
```

Expected output file (JSON):
- `results/rtp/rtp_continuations_harmformer.json`

Notes:
- With very small `--limit` values, it's normal to get `0.0%` leakage (which can make the RTP bars look “missing” in the plot). For a meaningful estimate, use a larger sample size (GPU recommended).
- For a quick CPU-friendly sanity check that produces non-zero bars, you can judge `prompt_and_continuation` (this is **not** the same metric as continuation-only leakage):

```bash
python scripts/evaluate_rtp_continuations.py \
  --device cpu \
  --batch-size 16 \
  --limit 50 \
  --max-chars 256 \
  --judge-text prompt_and_continuation \
  --output results/rtp/rtp_continuations_harmformer_smoke.json
```

To regenerate the HAVOC vs RTP plot:

```bash
python scripts/plot_havoc_rtp_compare.py \
  --havoc-results-dir results/havoc \
  --rtp-results-json results/rtp/rtp_continuations_harmformer.json \
  --out havoc-rtp-compare.png
```

#### Table 8 (large-scale toxicity prevalence; **approximate**)

The paper’s Table 8 estimates toxicity prevalence on **1,000,000 samples** from webscale datasets (CommonCrawl, C4, FineWeb).
This repo does **not** ship those datasets; you must provide an input file. To fit a **$20 budget**, our prevalence script defaults to
**10,000 samples** (set `--limit 1000000` to match the paper’s sampling count).

Example (JSONL with a `text` field; uses the default `harmformer` setup to avoid per-sample API costs):

```bash
python scripts/estimate_toxicity_prevalence.py \
  --input-path /path/to/dataset.jsonl \
  --input-format jsonl \
  --text-field text \
  --output results/prevalence/prevalence_10k.json
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

#### Multilingual extension (translation + multilingual evaluation)

Optional translation (slow; regenerates TSVs under `data/TTP-Eval/translated/`):

```bash
sbatch jobs/run_translate_ttp_eval.sh
```

Multilingual evaluation (writes JSONs under `results/ttp_eval_multilingual/`):

```bash
sbatch jobs/run_multilingual_extension.sh
```

#### RTP extension (HarmFormer on RTP continuations)

Writes `results/rtp/rtp_continuations_harmformer.json`:

```bash
sbatch jobs/run_rtp_extension.sh
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

If you want an API TTP row, run:

```bash
sbatch jobs/run_baselines_api.sh
```

#### Table 7 (OpenAI Moderation test set): Llama Guard / TTP / HarmFormer (Perspective omitted)

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
