#!/usr/bin/env bash
#SBATCH --job-name=deepspec-dspark-eval
#SBATCH --partition=plm
#SBATCH --quotatype=reserved
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#SBATCH --no-requeue
#SBATCH --output=/mnt/hwfile/shaofanghao/MTS/3rd/DeepSpec/local_outputs/eval/slurm/%x-%j.out
#SBATCH --error=/mnt/hwfile/shaofanghao/MTS/3rd/DeepSpec/local_outputs/eval/slurm/%x-%j.err

# Examples:
#
#   sbatch 3rd/DeepSpec/scripts/eval/eval_dspark_local.sbatch
#
#   TASKS=alpaca:32,gsm8k:32 MAX_NEW_TOKENS=256 \
#     sbatch 3rd/DeepSpec/scripts/eval/eval_dspark_local.sbatch
#
# Notes:
# - This script uses local model directories by default and forces offline mode.
# - Use one GPU by default: DeepSpec eval spawns one process per visible GPU and
#   each process loads both target and draft models, so multi-GPU mainly splits
#   samples but duplicates model memory.

set -euo pipefail
if [[ -f "${HOME}/.shell_common" ]]; then
  source "${HOME}/.shell_common"
fi

REPO="${REPO:-/mnt/hwfile/shaofanghao/MTS}"
DEEPSPEC_ROOT="${DEEPSPEC_ROOT:-${REPO}/3rd/DeepSpec}"
PYTHON_BIN="${PYTHON_BIN:-${REPO}/.venv/bin/python}"
RUN_ID="${RUN_ID:-$(TZ=Asia/Shanghai date +%Y%m%d-%H%M%S)-dspark_decode_eval}"
SLURM_LOG_DIR="${SLURM_LOG_DIR:-${DEEPSPEC_ROOT}/local_outputs/eval/slurm}"

if [[ -z "${SLURM_JOB_ID:-}" && "${DEEPSPEC_EVAL_NO_SUBMIT:-0}" != "1" ]]; then
  mkdir -p "${SLURM_LOG_DIR}"
  exec sbatch \
    --output="${SLURM_LOG_DIR}/%x-%j.out" \
    --error="${SLURM_LOG_DIR}/%x-%j.err" \
    "$0" "$@"
fi

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
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

TARGET_MODEL="${TARGET_MODEL:-${REPO}/model/0000-base_models/qwen3base_4B}"
DRAFT_CKPT="${DRAFT_CKPT:-${REPO}/model/0000-base_models/dspark_qwen3_4b_block7}"
DATASET_ROOT="${DATASET_ROOT:-${DEEPSPEC_ROOT}/eval_datasets}"
TASKS="${TASKS:-alpaca:8}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"
TEMPERATURE="${TEMPERATURE:-0.0}"
CONFIDENCE_THRESHOLD="${CONFIDENCE_THRESHOLD:-0.0}"
TENSORBOARD_DIR="${TENSORBOARD_DIR:-${DEEPSPEC_ROOT}/local_outputs/eval/${RUN_ID}/tensorboard}"

if [[ ! -d "${TARGET_MODEL}" ]]; then
  echo "[dspark-eval][error] missing target model: ${TARGET_MODEL}" >&2
  exit 2
fi
if [[ ! -d "${DRAFT_CKPT}" ]]; then
  echo "[dspark-eval][error] missing DSpark draft checkpoint: ${DRAFT_CKPT}" >&2
  exit 2
fi

echo "[dspark-eval] run_id=${RUN_ID}"
echo "[dspark-eval] target=${TARGET_MODEL}"
echo "[dspark-eval] draft=${DRAFT_CKPT}"
echo "[dspark-eval] tasks=${TASKS}"
echo "[dspark-eval] max_new_tokens=${MAX_NEW_TOKENS}"
echo "[dspark-eval] tensorboard=${TENSORBOARD_DIR}"
echo "[dspark-eval] cuda_visible_devices=${CUDA_VISIBLE_DEVICES}"

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
