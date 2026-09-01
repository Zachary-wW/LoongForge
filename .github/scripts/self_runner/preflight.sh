#!/usr/bin/env bash
# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

suite="${1:-}"
build_image="${2:-false}"
[[ "$suite" =~ ^(llm_vlm|embodied)$ ]] || { printf '%s\n' 'suite: invalid' >&2; exit 2; }
[[ "$build_image" == true || "$build_image" == false ]] || {
  printf '%s\n' 'build_image: invalid' >&2
  exit 2
}

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck disable=SC1091
source "$script_dir/../load_ci_config.sh"

required=(
  LOONGFORGE_DEFAULT_IMAGE
  LOONGFORGE_HOST_DATA_ROOT
  LOONGFORGE_CONTAINER_DATA_ROOT
  LOONGFORGE_HOST_OUTPUT_ROOT
  LOONGFORGE_CONTAINER_OUTPUT_ROOT
  LOONGFORGE_CONTAINER_SOURCE
  LOONGFORGE_RUNNER_LOG_ROOT
  TRITON_LIBCUDA_PATH
  LOONGFORGE_GPU_DEVICE
)
for name in "${required[@]}"; do
  if [[ -z "${!name:-}" ]]; then
    printf '%s\n' "$name" >&2
    exit 2
  fi
done

for name in LOONGFORGE_HOST_DATA_ROOT LOONGFORGE_HOST_OUTPUT_ROOT LOONGFORGE_RUNNER_LOG_ROOT TRITON_LIBCUDA_PATH; do
  if [[ ! -e "${!name}" ]]; then
    printf '%s\n' 'mounts: missing' >&2
    exit 2
  fi
done

docker_bin="${DOCKER_BIN:-docker}"
if ! "$docker_bin" version >/dev/null 2>&1; then
  printf '%s\n' 'docker: missing' >&2
  exit 1
fi
printf '%s\n' 'docker: ok'

if [[ "$build_image" == true ]]; then
  if [[ "${LOONGFORGE_ALLOW_PR_IMAGE_BUILD:-false}" != true ]]; then
    printf '%s\n' 'build_image: disabled' >&2
    exit 2
  fi
  if ! "$docker_bin" buildx version >/dev/null 2>&1; then
    printf '%s\n' 'buildx: missing' >&2
    exit 1
  fi
  printf '%s\n' 'buildx: ok'
fi

if ! "$docker_bin" run --rm --device="$LOONGFORGE_GPU_DEVICE" "$LOONGFORGE_DEFAULT_IMAGE" true >/dev/null 2>&1; then
  printf '%s\n' 'image-device: invalid' >&2
  exit 1
fi
printf '%s\n' 'image-device: ok'

# Shared runners host other tenants' containers; fail fast with an
# actionable message when any GPU has less free memory than the suite
# needs, instead of burning a full training run on a late OOM. Set
# LOONGFORGE_MIN_FREE_GPU_MB (MiB per GPU) in the runner config to
# override; an empty value disables the check.
if [[ -v LOONGFORGE_MIN_FREE_GPU_MB ]]; then
  # An explicitly empty value is the documented way to disable this optional
  # host-occupancy check; only an unset variable receives the default.
  min_free_mb="$LOONGFORGE_MIN_FREE_GPU_MB"
else
  min_free_mb=60000
fi
if [[ -n "${LOONGFORGE_MIN_FREE_GPU_MB:-}" && ! "$min_free_mb" =~ ^[0-9]+$ ]]; then
  printf '%s\n' "gpu-memory: LOONGFORGE_MIN_FREE_GPU_MB must be a number in MiB or empty (got: ${LOONGFORGE_MIN_FREE_GPU_MB})." >&2
  exit 2
fi
if [[ "$min_free_mb" =~ ^[0-9]+$ ]] && command -v nvidia-smi >/dev/null 2>&1; then
  if ! free_line=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null); then
    printf '%s\n' 'gpu-memory: query failed (skipping check)' >&2
  elif [[ -z "${free_line//$'\n'/}" ]]; then
    # nvidia-smi exists but reports no GPUs at all; a GPU suite runner in
    # that state cannot train. Fail closed instead of passing vacuously.
    printf '%s\n' 'gpu-memory: nvidia-smi reported no GPUs.' >&2
    exit 1
  else
    deficit=""
    index=0
    while IFS= read -r free_mb; do
      # nvidia-smi reports "[N/A]" or "[Not Supported]" for GPUs in an
      # error or uninitialized state; treat those as unavailable rather
      # than letting arithmetic coercion pass the check (fail closed).
      if [[ ! "$free_mb" =~ ^[0-9]+$ ]]; then
        deficit="${deficit} gpu${index}=unavailable"
      elif (( free_mb < min_free_mb )); then
        deficit="${deficit} gpu${index}=${free_mb}MiB"
      fi
      index=$((index + 1))
    done <<<"$free_line"
    if [[ -n "$deficit" ]]; then
      printf '%s\n' "gpu-memory: insufficient (need >= ${min_free_mb}MiB per GPU; short:${deficit})." >&2
      printf '%s\n' 'The suite runner is shared; retry later or free the GPUs listed above.' >&2
      exit 1
    fi
  fi
  printf '%s\n' "gpu-memory: ok (>= ${min_free_mb}MiB free per GPU)"
fi
printf '%s\n' 'mounts: ok'
