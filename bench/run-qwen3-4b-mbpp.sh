#!/bin/bash
# run-qwen3-4b-mbpp.sh
# Benchmark script for prime-rl on MBPP using 2 train + 6 infer GPUs

pkill -9 python
sleep 3

set -ex

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
PRIME_RL_DIR="${SCRIPT_DIR}/.."

cd "${PRIME_RL_DIR}"

# 1. Install custom environment via uv
uv pip install -e "${PRIME_RL_DIR}/benchmark_envs/mbpp_env"

# 2. Run the unified RL pipeline
export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

# Launch prime-rl pipeline
uv run rl @ "${SCRIPT_DIR}/rl.toml"
