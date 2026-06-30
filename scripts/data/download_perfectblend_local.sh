#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEEPSPEC_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
MTS_ROOT="$(cd "${DEEPSPEC_ROOT}/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${MTS_ROOT}/.venv/bin/python}"

DATA_ROOT="${DATA_ROOT:-${DEEPSPEC_ROOT}/local_data/perfectblend}"
HF_HOME="${HF_HOME:-${DEEPSPEC_ROOT}/local_data/hf_home}"
HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${DEEPSPEC_ROOT}/local_data/hf_datasets}"
HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

DATASET_NAME="${DATASET_NAME:-mlabonne/open-perfectblend}"
SAMPLE_SIZE="${SAMPLE_SIZE:-}"
TEST_SIZE="${TEST_SIZE:-0.05}"
SEED="${SEED:-42}"

mkdir -p "${DATA_ROOT}" "${HF_HOME}" "${HF_DATASETS_CACHE}"
cd "${DEEPSPEC_ROOT}"

unset http_proxy
unset https_proxy
unset HTTP_PROXY
unset HTTPS_PROXY
unset all_proxy
unset ALL_PROXY

args=(
  scripts/data/download_and_split.py
  --dataset-name "${DATASET_NAME}"
  --test-size "${TEST_SIZE}"
  --seed "${SEED}"
  --train-output-path "${DATA_ROOT}/perfectblend_train.jsonl"
  --test-output-dir "${DATA_ROOT}/eval_datasets"
  --skip-invalid
  --skip-existing
)

if [[ -n "${SAMPLE_SIZE}" ]]; then
  args+=(--sample-size "${SAMPLE_SIZE}")
fi

HF_ENDPOINT="${HF_ENDPOINT}" \
HF_HOME="${HF_HOME}" \
HF_DATASETS_CACHE="${HF_DATASETS_CACHE}" \
"${PYTHON_BIN}" "${args[@]}"
