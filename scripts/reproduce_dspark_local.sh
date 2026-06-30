#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEEPSPEC_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
MTS_ROOT="$(cd "${DEEPSPEC_ROOT}/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${MTS_ROOT}/.venv/bin/python}"
CONFIG_PATH="${CONFIG_PATH:-config/dspark/dspark_qwen3base_4b_local.py}"
TARGET_MODEL="${TARGET_MODEL:-${MTS_ROOT}/model/0000-base_models/qwen3base_4B}"
TARGET_CACHE_DIR="${TARGET_CACHE_DIR:-${DEEPSPEC_ROOT}/local_data/qwen3base_4b_target_cache}"
DRAFT_CKPT="${DRAFT_CKPT:-${DEEPSPEC_ROOT}/local_outputs/checkpoints/deepspec_local/dspark_block8_qwen3base_4b/step_latest}"
DATASET_ROOT="${DATASET_ROOT:-${DEEPSPEC_ROOT}/eval_datasets}"
TASKS="${TASKS:-alpaca:8}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"
TEMPERATURE="${TEMPERATURE:-0.0}"
CONFIDENCE_THRESHOLD="${CONFIDENCE_THRESHOLD:-0.0}"
MODE="${1:-smoke}"

cd "${DEEPSPEC_ROOT}"
export PYTHONPATH="${DEEPSPEC_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-29500}"
export RANK="${RANK:-0}"
export WORLD_SIZE="${WORLD_SIZE:-1}"

case "${MODE}" in
  smoke)
    "${PYTHON_BIN}" scripts/smoke_dspark_minimal.py
    ;;
  decode-smoke)
    "${PYTHON_BIN}" scripts/smoke_dspark_decode.py
    ;;
  train)
    if [[ ! -d "${TARGET_MODEL}" ]]; then
      echo "missing target model: ${TARGET_MODEL}" >&2
      exit 2
    fi
    if [[ ! -d "${TARGET_CACHE_DIR}" ]]; then
      echo "missing target cache: ${TARGET_CACHE_DIR}" >&2
      echo "prepare it first with scripts/data/prepare_target_cache.py; the full cache can be very large." >&2
      exit 2
    fi
    "${PYTHON_BIN}" train.py \
      --config "${CONFIG_PATH}" \
      --opts "model.target_model_name_or_path=${TARGET_MODEL}" \
      --opts "data.target_cache_path=${TARGET_CACHE_DIR}"
    ;;
  eval|decode)
    if [[ ! -d "${TARGET_MODEL}" ]]; then
      echo "missing target model: ${TARGET_MODEL}" >&2
      exit 2
    fi
    if [[ ! -d "${DRAFT_CKPT}" ]]; then
      echo "missing draft checkpoint: ${DRAFT_CKPT}" >&2
      exit 2
    fi
    "${PYTHON_BIN}" eval.py \
      --target_name_or_path "${TARGET_MODEL}" \
      --draft_name_or_path "${DRAFT_CKPT}" \
      --dataset-root "${DATASET_ROOT}" \
      --tasks "${TASKS}" \
      --max-new-tokens "${MAX_NEW_TOKENS}" \
      --temperature "${TEMPERATURE}" \
      --confidence-threshold "${CONFIDENCE_THRESHOLD}"
    ;;
  *)
    echo "usage: $0 [smoke|decode-smoke|train|eval|decode]" >&2
    exit 2
    ;;
esac
