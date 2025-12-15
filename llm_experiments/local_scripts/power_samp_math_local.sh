#!/bin/bash
set -euo pipefail

# Runs power_samp_math.py locally without sbatch/salloc.
# Usage:
#   ./power_samp_math_local.sh                 # run all shard/seed combos sequentially
#   ./power_samp_math_local.sh 2 5             # run a single shard/seed
# Environment overrides:
#   CONDA_ENV (default: cloudspace)
#   NUM_SHARDS (default: 5)
#   NUM_SEEDS (default: 8)
#   MODEL_NAME (default: qwen_math)
#   MCMC_STEPS (default: 10)
#   TEMP (default: 0.25)
#   DYNAMIC_METRIC (default: none; entropy/perplexity/self_confidence; not supported by power_samp_math.py)
#   ENTROPY_THRESHOLD (default: 1.0)
#   PERPLEXITY_THRESHOLD (default: 3.0)
#   SELF_CONF_THRESHOLD (default: 0.8)
#   DYNAMIC_MIN_TOKENS (default: 64)
#   RUN_MODE (default: all; baseline_only/mcmc_only)
#   BASELINE_VARIANT (default: both; naive/std)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LLM_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
CONDA_ENV="${CONDA_ENV:-cloudspace}"
NUM_SHARDS="${NUM_SHARDS:-5}"
NUM_SEEDS="${NUM_SEEDS:-8}"
MODEL_NAME="${MODEL_NAME:-qwen_math}"
MCMC_STEPS="${MCMC_STEPS:-10}"
TEMP="${TEMP:-0.25}"
DYNAMIC_METRIC="${DYNAMIC_METRIC:-none}"
ENTROPY_THRESHOLD="${ENTROPY_THRESHOLD:-1.0}"
PERPLEXITY_THRESHOLD="${PERPLEXITY_THRESHOLD:-3.0}"
SELF_CONF_THRESHOLD="${SELF_CONF_THRESHOLD:-0.8}"
DYNAMIC_MIN_TOKENS="${DYNAMIC_MIN_TOKENS:-64}"
RUN_MODE="${RUN_MODE:-all}"
BASELINE_VARIANT="${BASELINE_VARIANT:-both}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

cd "${LLM_DIR}"

run_job() {
  local batch_idx="$1"
  local seed="$2"
  shift 2
  local extra_args=("$@")
  echo "Running shard BATCH_IDX=${batch_idx} with SEED=${seed}"
  python power_samp_math.py \
    --batch_idx="${batch_idx}" \
    --mcmc_steps="${MCMC_STEPS}" \
    --temp="${TEMP}" \
    --seed="${seed}" \
    --model="${MODEL_NAME}" \
    --run_mode="${RUN_MODE}" \
    --baseline_variant="${BASELINE_VARIANT}" \
    --dynamic_metric="${DYNAMIC_METRIC}" \
    --entropy_threshold="${ENTROPY_THRESHOLD}" \
    --perplexity_threshold="${PERPLEXITY_THRESHOLD}" \
    --self_conf_threshold="${SELF_CONF_THRESHOLD}" \
    --dynamic_min_tokens="${DYNAMIC_MIN_TOKENS}" \
    "${extra_args[@]}"
}

EXTRA_ARGS=()
if [[ $# -ge 2 ]]; then
  EXTRA_ARGS=("${@:3}")
  run_job "$1" "$2" "${EXTRA_ARGS[@]}"
elif [[ $# -eq 0 ]]; then
  for batch_idx in $(seq 0 $((NUM_SHARDS - 1))); do
    for seed in $(seq 0 $((NUM_SEEDS - 1))); do
      run_job "${batch_idx}" "${seed}" "${EXTRA_ARGS[@]}"
    done
  done
else
  echo "Usage: $0 [BATCH_IDX SEED]" >&2
  exit 1
fi
