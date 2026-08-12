#!/usr/bin/env bash
# ============================================================================
# RAEv2 Stage 1 train launcher (decoder + LPIPS + adversarial loss).
# Usage: bash scripts/raev2/stage1/train.sh <config.yaml> <gpu_ids> [master_port] [extra_args...]
# Optional env: FRESH_START | RESUME_CKPT | RESULTS_DIR | EXP_NAME | EXPERIMENT_NAME
# Example:
#   bash scripts/raev2/stage1/train.sh \
#     configs/raev2/stage1/train/codec_adaptation/dinov3l_k7_mix2p5m_ft20.yaml 0,1,2,3
# ============================================================================
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../../.." && pwd)"
CONFIG=${1:?Usage: bash scripts/raev2/stage1/train.sh <config_path> <gpu_ids> [master_port]}
GPUS=${2:?Usage: bash scripts/raev2/stage1/train.sh <config_path> <gpu_ids> [master_port]}
PORT=${3:-29610}
EXTRA_ARGS=("${@:4}")
PYTHON_BIN=${PYTHON_BIN:-python}

cd "$ROOT_DIR"
[ ! -f "$CONFIG" ] && { echo "config not found: $CONFIG"; exit 1; }

export PYTHONPATH="$ROOT_DIR:$ROOT_DIR/src:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

NP=$(($(echo "$GPUS" | tr -cd , | wc -c)+1))
RESULTS=${RESULTS_DIR:-experiments/raev2/stage1}
# EXP_NAME defaults from config path:
# configs/raev2/stage1/train/<group>/<experiment>.yaml -> <group>/<experiment>
DEFAULT_EXP_NAME="$(echo "$CONFIG" | sed -E 's|^.*train/||; s|\.yaml$||')"
EXP_NAME="${EXP_NAME:-$DEFAULT_EXP_NAME}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-$EXP_NAME}"
mkdir -p "$RESULTS"

RESUME_ARGS=()
[[ -n "${RESUME_CKPT:-}" ]] && RESUME_ARGS+=(--resume "$RESUME_CKPT")

echo "[raev2_stage1_train] cfg=$CONFIG nproc=$NP port=$PORT exp=$EXPERIMENT_NAME resume=${RESUME_CKPT:-<auto>}"
CUDA_VISIBLE_DEVICES="$GPUS" "$PYTHON_BIN" -m torch.distributed.run \
  --standalone --nproc_per_node="$NP" --master_port="$PORT" \
  tools/raev2/train_stage1.py \
  --config "$CONFIG" \
  --results-dir "$RESULTS" \
  "${RESUME_ARGS[@]}" \
  "${EXTRA_ARGS[@]}"
# Precision, evaluation, and checkpoint policy are yaml-owned.
