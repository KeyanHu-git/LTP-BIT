#!/usr/bin/env bash
# RAEv2 Stage-1 config-driven reconstruction.
# Usage: bash scripts/raev2/stage1/test.sh <config.yaml> <gpu_ids> [master_port]
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../../.." && pwd)"
CONFIG=${1:?Usage: bash scripts/raev2/stage1/test.sh <config_path> <gpu_ids> [master_port]}
GPUS=${2:?Usage: bash scripts/raev2/stage1/test.sh <config_path> <gpu_ids> [master_port]}
PORT=${3:-29620}
PYTHON_BIN=${PYTHON_BIN:-python}

cd "$ROOT_DIR"
[ ! -f "$CONFIG" ] && { echo "config not found: $CONFIG"; exit 1; }

export PYTHONPATH="$ROOT_DIR:$ROOT_DIR/src:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

NP=$(($(echo "$GPUS" | tr -cd , | wc -c)+1))

echo "[raev2_stage1_test] cfg=$CONFIG nproc=$NP port=$PORT"
CUDA_VISIBLE_DEVICES="$GPUS" "$PYTHON_BIN" -m torch.distributed.run \
  --standalone --nproc_per_node="$NP" --master_port="$PORT" \
  tools/raev2/test_stage1.py \
  --config "$CONFIG"
