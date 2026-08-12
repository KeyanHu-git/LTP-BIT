#!/usr/bin/env bash
# ============================================================================
# RAEv2 Stage 3 train launcher (paired i2i fine-tuning).
# Usage: bash scripts/raev2/stage3/train.sh <config.yaml> <gpu_ids> [master_port]
# Optional env: INIT_CKPT | RESUME_CKPT | FRESH_START | EXTRA_ARGS | RESULTS_DIR | EXP_NAME | EXPERIMENT_NAME
#               STAGE3_AUTO_EVAL | EVAL_CONFIG | EVAL_PORT | EVAL_EXTRA_ARGS
# ============================================================================
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../../.." && pwd)"
CONFIG=${1:?Usage: bash scripts/raev2/stage3/train.sh <config_path> <gpu_ids> [master_port]}
GPUS=${2:?Usage: bash scripts/raev2/stage3/train.sh <config_path> <gpu_ids> [master_port]}
PORT=${3:-29650}
PYTHON_BIN=${PYTHON_BIN:-python}

cd "$ROOT_DIR"
[ ! -f "$CONFIG" ] && { echo "config not found: $CONFIG"; exit 1; }

export PYTHONPATH="$ROOT_DIR:$ROOT_DIR/src:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

NP=$(($(echo "$GPUS" | tr -cd , | wc -c)+1))
RESULTS=${RESULTS_DIR:-experiments/raev2/stage3}
DEFAULT_EXP_NAME="$(echo "$CONFIG" | sed -E 's|^.*train/||; s|\.yaml$||')"
EXP_NAME="${EXP_NAME:-$DEFAULT_EXP_NAME}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-$EXP_NAME}"
# Logs belong to the run, not the results root: write into the experiment dir
# ($RESULTS/$EXPERIMENT_NAME) so bookkeeping never clutters the parent folder.
EXP_DIR="$RESULTS/$EXPERIMENT_NAME"
mkdir -p "$EXP_DIR"
RUN_LOG="$EXP_DIR/train_${PORT}_$(date +%Y%m%d_%H%M%S).log"

STAGE3_AUTO_EVAL=${STAGE3_AUTO_EVAL:-1}
if [ "$STAGE3_AUTO_EVAL" = "1" ]; then
  if [ -z "${EVAL_CONFIG:-}" ]; then
    TRAIN_ROOT="$(realpath "$ROOT_DIR/configs/raev2/stage3/train")"
    CONFIG_ABS="$(realpath "$CONFIG")"
    case "$CONFIG_ABS" in
      "$TRAIN_ROOT"/*.yaml)
        REL_CONFIG="${CONFIG_ABS#"$TRAIN_ROOT"/}"
        REL_DIR="$(dirname "$REL_CONFIG")"
        [ "$REL_DIR" = "." ] && REL_DIR=""
        EVAL_CONFIG="$ROOT_DIR/configs/raev2/stage3/test/${REL_DIR:+$REL_DIR/}test_$(basename "$REL_CONFIG")"
        ;;
      *)
        echo "cannot infer evaluation config outside $TRAIN_ROOT: $CONFIG" >&2
        exit 2
        ;;
    esac
  fi
  [ -f "$EVAL_CONFIG" ] || { echo "evaluation config not found: $EVAL_CONFIG" >&2; exit 2; }
fi

INIT_ARG=""
[ -n "${INIT_CKPT:-}" ] && INIT_ARG="--init-ckpt $INIT_CKPT"
RESUME_ARG=""
[ -n "${RESUME_CKPT:-}" ] && RESUME_ARG="--resume $RESUME_CKPT"

echo "[raev2_stage3_train] cfg=$CONFIG nproc=$NP port=$PORT exp=$EXPERIMENT_NAME init=${INIT_CKPT:-<config/cold>} resume=${RESUME_CKPT:-<auto>} log=$RUN_LOG"
CUDA_VISIBLE_DEVICES="$GPUS" "$PYTHON_BIN" -m torch.distributed.run \
  --standalone --nproc_per_node="$NP" --master_port="$PORT" \
  tools/raev2/train_stage3.py \
  --config "$CONFIG" \
  --results-dir "$RESULTS" \
  $INIT_ARG \
  $RESUME_ARG \
  ${EXTRA_ARGS:-} 2>&1 | tee -a "$RUN_LOG"

if [ "$STAGE3_AUTO_EVAL" = "1" ]; then
  EVAL_PORT=${EVAL_PORT:-$((PORT + 1))}
  EVAL_LOG="$EXP_DIR/eval_${EVAL_PORT}_$(date +%Y%m%d_%H%M%S).log"
  [ -f "$EXP_DIR/checkpoints/ep-last.pt" ] || { echo "evaluation checkpoint not found: $EXP_DIR/checkpoints/ep-last.pt" >&2; exit 2; }
  echo "[raev2_stage3_train] training complete; starting evaluation cfg=$EVAL_CONFIG port=$EVAL_PORT log=$EVAL_LOG"
  CKPT="$EXP_DIR/checkpoints/ep-last.pt" \
  SAMPLE_DIR="$EXP_DIR" \
  SAVE_FOLDER="inference/full" \
  EXTRA_ARGS="${EVAL_EXTRA_ARGS:-}" \
    bash scripts/raev2/stage3/test.sh "$EVAL_CONFIG" "$GPUS" "$EVAL_PORT" 2>&1 | tee -a "$EVAL_LOG"
fi
