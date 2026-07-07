#!/bin/bash
#SBATCH --job-name=drift_confirmation
#SBATCH --partition=rome
#SBATCH --time=08:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/drift_confirmation_%j.out
#SBATCH --error=logs/drift_confirmation_%j.err

# Reviewer item 1 (+4): direct snapshot evidence for the gpt-4o drift claim.
#
# Default is MINIMAL mode — the cheapest run that satisfies the reviewer
# (~2 passes ~= $13 total at ~2.3M tokens/pass):
#   * TTP-Eval on the floating `gpt-4o` alias + ONE pinned snapshot, chosen by
#     a ~$0.01 fingerprint probe (a cheap-tier snapshot whose fingerprint
#     differs from the floating alias), seeded, fingerprinted per call;
#   * the --invalid-policy sensitivity (item 4) recomputed OFFLINE from the
#     floating run's per-sample data — exact, zero API cost
#     (scripts/recompute_invalid_sensitivity.py);
#   * paste-ready numbers via summarize_drift_confirmation.py.
#
# FULL=1 adds (~7 more passes, ~$55–70): all three pinned snapshots
# (2024-05-13 is 2x price), API-measured sensitivity passes, and the 3x
# same-snapshot noise floor. Corroboration only — not reviewer-required.
#
# Works on Slurm (sbatch) OR locally (bash jobs/run_drift_confirmation.sh) — the
# `module` block is guarded. CPU/API only; no GPU. Est. ~$8 per pass (n=393).
#
# Env knobs (all optional):
#   FULL=1      run the full corroboration suite (default: minimal)
#   SNAPSHOTS   space-separated dated snapshot ids (overrides mode default)
#   NOISE_REPEATS  passes for the noise floor in FULL mode (default 3)
#   SEED        decoding seed (default 12345)

set -euo pipefail
mkdir -p logs

if command -v module >/dev/null 2>&1; then
  module purge || true
  module load 2023 || true
  module load Python/3.11.3-GCCcore-12.3.0 || true
fi

PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$PROJECT_DIR"

# Activate venv if one exists (Snellius: ./venv; local: ./venv too).
if [ -f venv/bin/activate ]; then
  # shellcheck disable=SC1091
  source venv/bin/activate
fi

# Load credentials from .env (repo-local, then $HOME fallback for Slurm).
for envf in "$PROJECT_DIR/.env" "$HOME/TowardsSaferPretraining/.env"; do
  if [ -f "$envf" ]; then set -a; . "$envf"; set +a; break; fi
done

SEED="${SEED:-12345}"
FULL="${FULL:-0}"
NOISE_REPEATS="${NOISE_REPEATS:-3}"
OUT_DIR="results/ttp_eval_drift_confirmation"
mkdir -p "$OUT_DIR" results/ttp_eval_noise_floor results/codecarbon
export CODECARBON_OUTPUT_DIR="${CODECARBON_OUTPUT_DIR:-$PROJECT_DIR/results/codecarbon}"

# Pick the endpoint that yields the richest fingerprint evidence: OpenAI direct
# returns system_fingerprint; OpenRouter frequently does not.
have() { [ -n "${1:-}" ] && ! printf '%s' "$1" | grep -qiE 'YOUR|HERE'; }
if have "${OPENAI_API_KEY:-}"; then
  SETUP="openai_ttp"; MODEL_FLAG="--openai-model"; FLOATING="gpt-4o"; PFX=""
elif have "${OPENROUTER_API_KEY:-}"; then
  SETUP="openrouter_ttp"; MODEL_FLAG="--openrouter-model"; FLOATING="openai/gpt-4o"; PFX="openai/"
else
  echo "Error: set OPENAI_API_KEY (preferred, for fingerprints) or OPENROUTER_API_KEY in .env" >&2
  exit 1
fi
# Snapshot choice. NOTE pricing: gpt-4o-2024-05-13 is 2x the others
# ($5/$15 per M vs $2.50/$10); at ~2.3M tokens/pass that's ~$12.5 vs ~$6.5.
if [ "$FULL" = "1" ]; then
  DEFAULT_SNAPSHOTS="${PFX}gpt-4o-2024-05-13 ${PFX}gpt-4o-2024-08-06 ${PFX}gpt-4o-2024-11-20"
else
  DEFAULT_SNAPSHOTS="__probe__"   # minimal: pick one cheap snapshot via fingerprint probe below
fi
# shellcheck disable=SC2206
SNAPSHOTS=(${SNAPSHOTS:-$DEFAULT_SNAPSHOTS})
echo "Using setup=$SETUP  floating=$FLOATING"
echo "Mode: $([ "$FULL" = "1" ] && echo FULL || echo "MINIMAL (~2 passes, ~\$13 at ~2.3M tok/pass)")"

# shellcheck disable=SC1091
source "$PROJECT_DIR/jobs/_wandb_args.sh"
build_wandb_args ttp-eval drift snapshot

# Preflight + fingerprint probe: 1-token pings (~$0.01 total) that (a) verify
# the key works and has credit BEFORE any full pass, and (b) reveal which
# backend snapshot each alias serves. In minimal mode this picks the pinned
# snapshot to buy: the cheapest one whose fingerprint DIFFERS from the floating
# alias, so the single paid contrast is guaranteed not to be a duplicate.
python - "$SETUP" "$FLOATING" "$PFX" "$OUT_DIR/.chosen_snapshot" <<'PY'
import os, sys
setup, floating, pfx, chosen_path = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
from openai import OpenAI
if setup == "openai_ttp":
    client = OpenAI()
else:
    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.environ["OPENROUTER_API_KEY"])

# Cheap-first candidate order: 08-06 / 11-20 are $2.50/M in; 05-13 is $5/M.
candidates = [pfx + s for s in ("gpt-4o-2024-08-06", "gpt-4o-2024-11-20", "gpt-4o-2024-05-13")]
fps = {}
for model in [floating] + candidates:
    try:
        r = client.chat.completions.create(model=model, messages=[{"role": "user", "content": "ping"}], max_tokens=1)
        fps[model] = getattr(r, "system_fingerprint", None)
    except Exception as e:
        if model == floating:
            sys.exit(f"Preflight call failed ({type(e).__name__}): {e}\nFix key/credits before burning a full pass.")
        fps[model] = f"<unavailable: {type(e).__name__}>"
print("Fingerprint probe:")
for m, fp in fps.items():
    print(f"  {m:<28} {fp}")

float_fp = fps[floating]
usable = [m for m in candidates if not str(fps[m]).startswith("<unavailable")]
informative = float_fp and any(fps[m] and fps[m] != float_fp for m in usable)
if informative:
    chosen = next(m for m in usable if fps[m] and fps[m] != float_fp)
    print(f"Chosen pinned snapshot: {chosen} (fingerprint differs from floating alias)")
else:
    # Null/constant fingerprints can no longer distinguish snapshots; default
    # to 2024-08-06 — the snapshot the alias documentedly served during the
    # April/May drift window, i.e. the scientifically relevant contrast.
    chosen = usable[0] if usable else candidates[0]
    print(f"Fingerprints uninformative; defaulting to {chosen}")
open(chosen_path, "w").write(chosen)
PY

run_eval() {  # <model> <output> [extra args...]
  local model="$1"; local out="$2"; shift 2
  python scripts/evaluate_ttp_eval.py \
    --data-path data/TTP-Eval/TTPEval.tsv \
    --setups "$SETUP" "$MODEL_FLAG" "$model" \
    --dimension toxic --seed "$SEED" \
    --output "$out" "$@" ${WANDB_ARGS[@]+"${WANDB_ARGS[@]}"}
  # Fail fast if the leg was truncated (e.g. credits ran out mid-run): a
  # quota-dead leg still "succeeds" per-sample-excluded, which would silently
  # poison every later leg and the summary.
  python - "$out" <<'PY'
import json, sys
payload = json.load(open(sys.argv[1]))
for r in payload.get("results", []):
    n, failed = r.get("evaluated_samples") or 0, r.get("failed_samples") or 0
    total = n + failed
    if total and failed / total > 0.05:
        sys.exit(f"ABORT: {failed}/{total} samples failed in {sys.argv[1]} "
                 f"(likely quota/auth). Refusing to continue with truncated data.")
print(f"  leg OK: {sys.argv[1]}")
PY
}

tag() { printf '%s' "$1" | tr '/:.' '___'; }

# Resolve the probe's snapshot choice (minimal mode only).
if [ "${SNAPSHOTS[0]}" = "__probe__" ]; then
  SNAPSHOTS=("$(cat "$OUT_DIR/.chosen_snapshot")")
fi
echo "Snapshots to run: ${SNAPSHOTS[*]}"

# 1) Floating alias + each pinned snapshot (default --invalid-policy exclude).
run_eval "$FLOATING" "$OUT_DIR/floating_$(tag "$FLOATING").json" --invalid-policy exclude
for snap in "${SNAPSHOTS[@]}"; do
  run_eval "$snap" "$OUT_DIR/pinned_$(tag "$snap").json" --invalid-policy exclude
done

# 2) --invalid-policy sensitivity on the headline (floating) row (item 4).
if [ "$FULL" = "1" ]; then
  # FULL: measure the non-default policies with real API passes.
  cp "$OUT_DIR/floating_$(tag "$FLOATING").json" "$OUT_DIR/sensitivity_exclude.json"
  for pol in non_toxic toxic; do
    run_eval "$FLOATING" "$OUT_DIR/sensitivity_${pol}.json" --invalid-policy "$pol"
  done
else
  # MINIMAL: the three policies differ only in how failed samples are counted,
  # so recompute them exactly from the floating run's per-sample data. $0.
  python scripts/recompute_invalid_sensitivity.py \
    "$OUT_DIR/floating_$(tag "$FLOATING").json" --out-dir "$OUT_DIR"
fi

# 3) Same-snapshot noise floor on the floating alias (FULL only — corroboration).
if [ "$FULL" = "1" ]; then
  python scripts/ttp_noise_floor.py \
    --setup "$SETUP" "$MODEL_FLAG" "$FLOATING" \
    --repeats "$NOISE_REPEATS" --seed "$SEED" \
    --output "results/ttp_eval_noise_floor/${SETUP}_x${NOISE_REPEATS}.json" \
    ${WANDB_ARGS[@]+"${WANDB_ARGS[@]}"}
  # Copy the noise-floor JSON next to the others so the summary picks it up.
  cp "results/ttp_eval_noise_floor/${SETUP}_x${NOISE_REPEATS}.json" \
     "$OUT_DIR/${SETUP}_noise_floor.json" 2>/dev/null || true
fi

echo
echo "########## PASTE-READY SUMMARY ##########"
python scripts/summarize_drift_confirmation.py "$OUT_DIR"
