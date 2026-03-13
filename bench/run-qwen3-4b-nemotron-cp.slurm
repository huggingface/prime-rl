#!/bin/bash
#SBATCH --job-name=rl-nemotron-cp
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --partition=hopper-prod
#SBATCH --time=10:00:00
#SBATCH --output=/fsx/amine_dirhoussi/prime-rl/bench/logs/%x-%j.out
#SBATCH --error=/fsx/amine_dirhoussi/prime-rl/bench/logs/%x-%j.err

set -ex

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
PRIME_RL_DIR="${SCRIPT_DIR}/.."

cd "${PRIME_RL_DIR}"

export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

uv run rl @ "${SCRIPT_DIR}/rl_nemotron_cp.toml"
