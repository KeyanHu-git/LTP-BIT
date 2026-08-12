#!/usr/bin/env bash
# RAEv2 Stage-3 config-driven paired i2i evaluation.
# Usage: bash scripts/raev2/stage3/test.sh <config.yaml> <gpu_ids> [master_port]
# Optional env: CKPT | INIT_CKPT | SAMPLE_DIR | SAVE_FOLDER | EXTRA_ARGS
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../../.." && pwd)"
CONFIG=${1:?Usage: bash scripts/raev2/stage3/test.sh <config_path> <gpu_ids> [master_port]}
GPUS=${2:?Usage: bash scripts/raev2/stage3/test.sh <config_path> <gpu_ids> [master_port]}
PORT=${3:-29660}
PYTHON_BIN=${PYTHON_BIN:-python}

cd "$ROOT_DIR"
[ ! -f "$CONFIG" ] && { echo "config not found: $CONFIG"; exit 1; }

export PYTHONPATH="$ROOT_DIR:$ROOT_DIR/src:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

NP=$(($(echo "$GPUS" | tr -cd , | wc -c)+1))
CKPT_ARG=()
[ -n "${CKPT:-}" ] && CKPT_ARG=(--ckpt "$CKPT")
INIT_CKPT_ARG=()
[ -n "${INIT_CKPT:-}" ] && INIT_CKPT_ARG=(--init-ckpt "$INIT_CKPT")
SAMPLE_DIR_ARG=()
[ -n "${SAMPLE_DIR:-}" ] && SAMPLE_DIR_ARG=(--sample-dir "$SAMPLE_DIR")
SAVE_FOLDER_ARG=()
[ -n "${SAVE_FOLDER:-}" ] && SAVE_FOLDER_ARG=(--save-folder "$SAVE_FOLDER")

echo "[raev2_stage3_test] cfg=$CONFIG nproc=$NP port=$PORT"
CUDA_VISIBLE_DEVICES="$GPUS" "$PYTHON_BIN" -m torch.distributed.run \
  --standalone --nproc_per_node="$NP" --master_port="$PORT" \
  tools/raev2/test_stage3.py \
  --config "$CONFIG" \
  "${CKPT_ARG[@]}" \
  "${INIT_CKPT_ARG[@]}" \
  "${SAMPLE_DIR_ARG[@]}" \
  "${SAVE_FOLDER_ARG[@]}" \
  ${EXTRA_ARGS:-}
