#!/usr/bin/env bash
# ============================================================================
# RAEv2 Stage 2 train launcher (Image Generation / IG latent diffusion).
# Usage: bash scripts/raev2/stage2/train.sh <config.yaml> <gpu_ids> [master_port]
# Optional env: INIT_CKPT | RESUME_CKPT | EXTRA_ARGS | RESULTS_DIR | EXP_NAME | EXPERIMENT_NAME
# Example:
#   bash scripts/raev2/stage2/train.sh \
#     configs/raev2/stage2/train/git10m/igxl_1m_ep15.yaml 0,1,2,3
#   bash scripts/raev2/stage2/train.sh \
#     configs/raev2/stage2/train/git10m/igxl_1m_ep85_from_ep15.yaml 0,1,2,3
# ============================================================================
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../../.." && pwd)"
CONFIG=${1:?Usage: bash scripts/raev2/stage2/train.sh <config_path> <gpu_ids> [master_port]}
GPUS=${2:?Usage: bash scripts/raev2/stage2/train.sh <config_path> <gpu_ids> [master_port]}
PORT=${3:-29640}
PYTHON_BIN=${PYTHON_BIN:-python}

cd "$ROOT_DIR"
[ ! -f "$CONFIG" ] && { echo "config not found: $CONFIG"; exit 1; }

export PYTHONPATH="$ROOT_DIR:$ROOT_DIR/src:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

NP=$(($(echo "$GPUS" | tr -cd , | wc -c)+1))
RESULTS=${RESULTS_DIR:-experiments/raev2/stage2}
# EXP_NAME defaults from config path:
# configs/raev2/stage2/train/<family>/<dataset>/<experiment>.yaml
# -> <family>/<dataset>/<experiment>
DEFAULT_EXP_NAME="$(echo "$CONFIG" | sed -E 's|^.*train/||; s|\.yaml$||')"
EXP_NAME="${EXP_NAME:-$DEFAULT_EXP_NAME}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-$EXP_NAME}"
# Logs belong to the run, not the results root: write into the experiment dir
# ($RESULTS/$EXPERIMENT_NAME) so bookkeeping never clutters the parent folder.
EXP_DIR="$RESULTS/$EXPERIMENT_NAME"
mkdir -p "$EXP_DIR"
RUN_LOG="$EXP_DIR/train_${PORT}_$(date +%Y%m%d_%H%M%S).log"

INIT_ARG=""
[ -n "${INIT_CKPT:-}" ] && INIT_ARG="--init-ckpt $INIT_CKPT"
RESUME_ARG=""
[ -n "${RESUME_CKPT:-}" ] && RESUME_ARG="--resume $RESUME_CKPT"

echo "[raev2_stage2_train] cfg=$CONFIG nproc=$NP port=$PORT exp=$EXPERIMENT_NAME init=${INIT_CKPT:-<config/cold>} resume=${RESUME_CKPT:-<auto>} log=$RUN_LOG"
CUDA_VISIBLE_DEVICES="$GPUS" "$PYTHON_BIN" -m torch.distributed.run \
  --standalone --nproc_per_node="$NP" --master_port="$PORT" \
  tools/raev2/train_stage2.py \
  --config "$CONFIG" \
  --results-dir "$RESULTS" \
  $INIT_ARG \
  $RESUME_ARG \
  ${EXTRA_ARGS:-} 2>&1 | tee -a "$RUN_LOG"
# precision / optimizer / scheduler / eval / checkpoint policy are yaml-owned.
