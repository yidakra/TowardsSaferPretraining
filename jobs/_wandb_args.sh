# Shared helper for jobs/run_*.sh: builds the W&B argv suffix for evaluation
# scripts based on WANDB_* env vars. Source this file, then call:
#   build_wandb_args <tag1> [<tag2> ...]
# which populates the global WANDB_ARGS bash array. When WANDB_ENABLED!=1
# the array is left empty so callers can always splice it: "${WANDB_ARGS[@]}".

build_wandb_args() {
  WANDB_ARGS=()
  if [ "${WANDB_ENABLED:-0}" = "1" ]; then
    WANDB_ARGS+=(--wandb --wandb-project "${WANDB_PROJECT:-TowardsSaferPretraining}")
    if [ -n "${WANDB_ENTITY:-}" ]; then WANDB_ARGS+=(--wandb-entity "$WANDB_ENTITY"); fi
    if [ -n "${WANDB_GROUP:-}" ]; then WANDB_ARGS+=(--wandb-group "$WANDB_GROUP"); fi
    if [ -n "${WANDB_MODE:-}" ]; then WANDB_ARGS+=(--wandb-mode "$WANDB_MODE"); fi
    if [ "$#" -gt 0 ]; then WANDB_ARGS+=(--wandb-tags "$@"); fi
  fi
}
