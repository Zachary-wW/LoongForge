#!/usr/bin/env bash
# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail
umask 077

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
image=""
source_dir=""
sha=""
llm_model="deepseek_v2_lite"
vla_model="pi05_ddp"
master_port=$((20000 + ($$ % 30000)))

while [[ $# -gt 0 ]]; do
  case "$1" in
    --image) image="${2:-}"; shift 2 ;;
    --source) source_dir="${2:-}"; shift 2 ;;
    --sha) sha="${2:-}"; shift 2 ;;
    --llm-model) llm_model="${2:-}"; shift 2 ;;
    --vla-model) vla_model="${2:-}"; shift 2 ;;
    *) printf '%s\n' 'argument: invalid' >&2; exit 2 ;;
  esac
done

[[ "$image" =~ ^[A-Za-z0-9._/-]+:[A-Za-z0-9._-]+$ ]] || {
  printf '%s\n' 'image: invalid' >&2
  exit 2
}
[[ -d "$source_dir" && "$sha" =~ ^[0-9a-f]{40}$ ]] || {
  printf '%s\n' 'source and a full commit SHA are required' >&2
  exit 2
}
[[ "$llm_model" =~ ^[A-Za-z0-9][A-Za-z0-9._/-]*$ ]] || {
  printf '%s\n' 'llm model: invalid' >&2
  exit 2
}
[[ "$vla_model" =~ ^[A-Za-z0-9][A-Za-z0-9._/-]*$ ]] || {
  printf '%s\n' 'vla model: invalid' >&2
  exit 2
}

# shellcheck disable=SC1091
source "$script_dir/../load_ci_config.sh"

: "${LOONGFORGE_DEFAULT_IMAGE:?LOONGFORGE_DEFAULT_IMAGE is required}"
: "${LOONGFORGE_HOST_DATA_ROOT:?LOONGFORGE_HOST_DATA_ROOT is required}"
: "${LOONGFORGE_CONTAINER_DATA_ROOT:?LOONGFORGE_CONTAINER_DATA_ROOT is required}"
: "${LOONGFORGE_HOST_LLM_DATA_ROOT:?LOONGFORGE_HOST_LLM_DATA_ROOT is required}"
: "${LOONGFORGE_CONTAINER_LLM_DATA_ROOT:?LOONGFORGE_CONTAINER_LLM_DATA_ROOT is required}"
: "${LOONGFORGE_HOST_OUTPUT_ROOT:?LOONGFORGE_HOST_OUTPUT_ROOT is required}"
: "${LOONGFORGE_CONTAINER_OUTPUT_ROOT:?LOONGFORGE_CONTAINER_OUTPUT_ROOT is required}"
: "${LOONGFORGE_CONTAINER_SOURCE:?LOONGFORGE_CONTAINER_SOURCE is required}"
: "${LOONGFORGE_RUNNER_LOG_ROOT:?LOONGFORGE_RUNNER_LOG_ROOT is required}"

docker_bin="${DOCKER_BIN:-docker}"
artifact_dir="$PWD/loongforge-release-artifacts"
if [[ -L "$artifact_dir" || (-e "$artifact_dir" && ! -d "$artifact_dir") ]]; then
  printf '%s\n' 'artifact: invalid directory' >&2
  exit 2
fi
rm -rf "$artifact_dir"
mkdir -p "$artifact_dir" "$LOONGFORGE_RUNNER_LOG_ROOT" "$LOONGFORGE_HOST_OUTPUT_ROOT"

"$script_dir/preflight.sh" release false >/dev/null

container_name="loongforge-release-${sha:0:12}-$$"
llm_log="$LOONGFORGE_RUNNER_LOG_ROOT/${container_name}.llm.log"
vla_log="$LOONGFORGE_RUNNER_LOG_ROOT/${container_name}.vla.log"
result_raw="$artifact_dir/${container_name}.result.json"
status=1

redact_artifact() {
  local input="$1"
  local output="$2"
  if [[ -f "$input" ]]; then
    python3 "$script_dir/../../../ci/redact_ci_artifact.py" \
      --input "$input" --output "$output" || true
  fi
}

cleanup() {
  local exit_status="$status"
  set +e
  "$docker_bin" rm -f "$container_name" >/dev/null 2>&1 || true
  redact_artifact "$llm_log" "$artifact_dir/llm-regression.log"
  redact_artifact "$vla_log" "$artifact_dir/vla-regression.log"
  if [[ ! -f "$result_raw" ]]; then
    printf '{"status":"failed","suite":"release","models":"%s,%s","exit_code":%d}\n' \
      "$llm_model" "$vla_model" "$exit_status" >"$result_raw"
  fi
  redact_artifact "$result_raw" "$artifact_dir/release-regression.result.json"
  "$script_dir/cleanup.sh" >/dev/null 2>&1 || true
  if [[ -n "${GITHUB_OUTPUT:-}" ]]; then
    printf '%s\n' 'artifact_dir=loongforge-release-artifacts' >>"$GITHUB_OUTPUT"
  fi
  exit "$exit_status"
}
trap cleanup EXIT INT TERM

export LOONGFORGE_PULL_IMAGE=false
"$script_dir/create_container.sh" "$image" "$source_dir" release "$container_name" \
  >>"$llm_log" 2>&1

llm_model_q=$(printf '%q' "$llm_model")
llm_chip_q=$(printf '%q' "${LOONGFORGE_BASELINE_LLM_VLM:-a}")
container_source_q=$(printf '%q' "$LOONGFORGE_CONTAINER_SOURCE")
output_root_q=$(printf '%q' "$LOONGFORGE_CONTAINER_OUTPUT_ROOT")

set +e
"$docker_bin" exec \
  -e "PFS_PATH=$LOONGFORGE_CONTAINER_LLM_DATA_ROOT" \
  -e "TRAINING_LOG_PATH=$LOONGFORGE_CONTAINER_OUTPUT_ROOT/llm_vlm" \
  -e "RANK=0" \
  -e "WORLD_SIZE=1" \
  -e "MASTER_ADDR=127.0.0.1" \
  -e "MASTER_PORT=$master_port" \
  -e "LOONGFORGE_TEST_SUITE=llm_vlm" \
  "$container_name" \
  bash -lc \
  "cd ${container_source_q} && cd tests/llm_vlm && \
   python3 main.py --models ${llm_model_q} --chip ${llm_chip_q} \
   --tasks check_correctness_task check_precess_data_task \
   --training_type pretrain sft --node_nums 1 --gpu_nums 8 --check_loss_only" \
  >"$llm_log" 2>&1
llm_status=$?
set -e

vla_status=1
if [[ "$llm_status" -eq 0 ]]; then
  vla_model_q=$(printf '%q' "$vla_model")
  vla_chip_q=$(printf '%q' "${LOONGFORGE_BASELINE_EMBODIED:-p}")
  vla_data_q=$(printf '%q' "$LOONGFORGE_CONTAINER_DATA_ROOT")
  set +e
  "$docker_bin" exec \
    -e "LOCAL_VLA_ARTIFACTS_ROOT=$LOONGFORGE_CONTAINER_DATA_ROOT" \
    -e "EMBODIED_LOG_ROOT=$LOONGFORGE_CONTAINER_OUTPUT_ROOT/embodied" \
    -e "LOONGFORGE_TEST_SUITE=embodied" \
    "$container_name" \
    bash -lc \
    "cd ${container_source_q} && \
     LOCAL_VLA_ARTIFACTS_ROOT=${vla_data_q} \
     EMBODIED_LOG_ROOT=${output_root_q}/embodied \
     bash tests/embodied/run.sh --chip ${vla_chip_q} --models ${vla_model_q}" \
    >"$vla_log" 2>&1
  vla_status=$?
  set -e
fi

if [[ "$llm_status" -ne 0 ]]; then
  status="$llm_status"
elif [[ "$vla_status" -ne 0 ]]; then
  status="$vla_status"
else
  status=0
fi

printf '{"status":"%s","suite":"release","models":"%s,%s","exit_code":%d}\n' \
  "$([[ "$status" -eq 0 ]] && printf passed || printf failed)" \
  "$llm_model" "$vla_model" "$status" >"$result_raw"
exit "$status"
