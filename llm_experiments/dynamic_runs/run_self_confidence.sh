#!/bin/bash
set -euo pipefail

# Run any local sampling script with self-confidence-based dynamic stopping.
# Usage: ./run_self_confidence.sh [path_to_local_script] [script_args...]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_SCRIPT="$SCRIPT_DIR/../local_scripts/power_samp_math_local.sh"
TARGET_SCRIPT="${1:-$DEFAULT_SCRIPT}"
if [[ $# -gt 0 ]]; then
  shift
fi

export DYNAMIC_METRIC="self_confidence"
export ENTROPY_THRESHOLD="${ENTROPY_THRESHOLD:-1.0}"
export PERPLEXITY_THRESHOLD="${PERPLEXITY_THRESHOLD:-3.0}"
export SELF_CONF_THRESHOLD="${SELF_CONF_THRESHOLD:-0.85}"
export DYNAMIC_MIN_TOKENS="${DYNAMIC_MIN_TOKENS:-64}"

exec "${TARGET_SCRIPT}" "$@"
