#!/bin/bash
set -euo pipefail

# Runs power_samp_he.py locally without sbatch/salloc.
# Usage:
#   ./power_samp_he_local.sh              # run all shard/seed combos sequentially
#   ./power_samp_he_local.sh 0 7          # run a single shard/seed
# Environment overrides:
#   CONDA_ENV (default: cloudspace)
#   NUM_SHARDS (default: 4)
#   NUM_SEEDS (default: 8)
#   MODEL_NAME (default: qwen_math)
#   MCMC_STEPS (default: 10)
#   TEMP (default: 0.25)
#   DYNAMIC_METRIC (default: none)
#   ENTROPY_THRESHOLD (default: 1.0)
#   PERPLEXITY_THRESHOLD (default: 3.0)
#   SELF_CONF_THRESHOLD (default: 0.8)
#   DYNAMIC_MIN_TOKENS (default: 64)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LLM_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
CONDA_ENV="${CONDA_ENV:-cloudspace}"
NUM_SHARDS="${NUM_SHARDS:-4}"
NUM_SEEDS="${NUM_SEEDS:-8}"
MODEL_NAME="${MODEL_NAME:-qwen_math}"
MCMC_STEPS="${MCMC_STEPS:-10}"
TEMP="${TEMP:-0.25}"
DYNAMIC_METRIC="${DYNAMIC_METRIC:-none}"
ENTROPY_THRESHOLD="${ENTROPY_THRESHOLD:-1.0}"
PERPLEXITY_THRESHOLD="${PERPLEXITY_THRESHOLD:-3.0}"
SELF_CONF_THRESHOLD="${SELF_CONF_THRESHOLD:-0.8}"
DYNAMIC_MIN_TOKENS="${DYNAMIC_MIN_TOKENS:-64}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

cd "${LLM_DIR}"

run_job() {
  local batch_idx="$1"
  local seed="$2"
  echo "Running shard BATCH_IDX=${batch_idx} with SEED=${seed}"
  python power_samp_he.py \
    --batch_idx="${batch_idx}" \
    --mcmc_steps="${MCMC_STEPS}" \
    --temp="${TEMP}" \
    --seed="${seed}" \
    --model="${MODEL_NAME}" \
    --dynamic_metric="${DYNAMIC_METRIC}" \
    --entropy_threshold="${ENTROPY_THRESHOLD}" \
    --perplexity_threshold="${PERPLEXITY_THRESHOLD}" \
    --self_conf_threshold="${SELF_CONF_THRESHOLD}" \
    --dynamic_min_tokens="${DYNAMIC_MIN_TOKENS}"
}

if [[ $# -eq 2 ]]; then
  run_job "$1" "$2"
elif [[ $# -eq 0 ]]; then
  for batch_idx in $(seq 0 $((NUM_SHARDS - 1))); do
    for seed in $(seq 0 $((NUM_SEEDS - 1))); do
      run_job "${batch_idx}" "${seed}"
    done
  done
else
  echo "Usage: $0 [BATCH_IDX SEED]" >&2
  exit 1
fi
