#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEEPSPEC_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
MTS_ROOT="$(cd "${DEEPSPEC_ROOT}/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${MTS_ROOT}/.venv/bin/python}"

TARGET_MODEL="${TARGET_MODEL:-${MTS_ROOT}/model/0000-base_models/qwen3base_4B}"
DRAFT_CKPT="${DRAFT_CKPT:-${MTS_ROOT}/model/0000-base_models/dspark_qwen3_4b_block7}"
DATASET_ROOT="${DATASET_ROOT:-${DEEPSPEC_ROOT}/eval_datasets}"
TASKS="${TASKS:-alpaca:8}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"
TEMPERATURE="${TEMPERATURE:-0.0}"
CONFIDENCE_THRESHOLD="${CONFIDENCE_THRESHOLD:-0.0}"
RUN_NAME="${RUN_NAME:-dspark_qwen3_4b_block7_local}"
TENSORBOARD_DIR="${TENSORBOARD_DIR:-${DEEPSPEC_ROOT}/local_outputs/eval/${RUN_NAME}/tensorboard}"

cd "${DEEPSPEC_ROOT}"
export PYTHONPATH="${DEEPSPEC_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-29500}"
export RANK="${RANK:-0}"
export WORLD_SIZE="${WORLD_SIZE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

if [[ ! -d "${TARGET_MODEL}" ]]; then
  echo "missing target model: ${TARGET_MODEL}" >&2
  exit 2
fi
if [[ ! -d "${DRAFT_CKPT}" ]]; then
  echo "missing DSpark draft checkpoint: ${DRAFT_CKPT}" >&2
  exit 2
fi

"${PYTHON_BIN}" eval.py \
  --target_name_or_path "${TARGET_MODEL}" \
  --draft_name_or_path "${DRAFT_CKPT}" \
  --dataset-root "${DATASET_ROOT}" \
  --tasks "${TASKS}" \
  --max-new-tokens "${MAX_NEW_TOKENS}" \
  --temperature "${TEMPERATURE}" \
  --confidence-threshold "${CONFIDENCE_THRESHOLD}" \
  --tensorboard-dir "${TENSORBOARD_DIR}" \
  --step 0
