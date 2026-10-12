#!/usr/bin/env bash
# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

# Route release regression to the resolved Execution Slot. Remote execution
# synchronizes the checked-out source, runs the regression there, and returns
# only the portable release evidence to the Control Runner.

set -euo pipefail

image=""
source_dir=""
sha=""
llm_model="deepseek_v2_lite"
vla_model="pi05_ddp"

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

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
mode="${LOONGFORGE_EXECUTION_MODE:?execution slot is not resolved}"

if [[ "$mode" == local ]]; then
  exec "$script_dir/run_release_regression.sh" \
    --image "$image" \
    --source "$source_dir" \
    --sha "$sha" \
    --llm-model "$llm_model" \
    --vla-model "$vla_model"
fi

[[ "$mode" == ssh ]] || { printf '%s\n' 'execution-slot: invalid' >&2; exit 2; }
alias="${LOONGFORGE_EXECUTION_ALIAS:?execution slot is incomplete}"
root="${LOONGFORGE_EXECUTION_ROOT:?execution slot is incomplete}"
remote_source="${LOONGFORGE_EXECUTION_SOURCE:?execution slot is incomplete}"
remote_config="${LOONGFORGE_EXECUTION_CONFIG:?execution slot is incomplete}"
[[ "$alias" =~ ^[A-Za-z0-9._@-]+$ ]] || {
  printf '%s\n' 'execution-slot: invalid configuration' >&2
  exit 2
}
for path_name in root remote_source remote_config; do
  [[ "${!path_name}" =~ ^/[A-Za-z0-9._/-]+$ ]] || {
    printf '%s\n' 'execution-slot: invalid configuration' >&2
    exit 2
  }
done
run_id="${GITHUB_RUN_ID:-manual-$(date -u +%Y%m%d%H%M%S)}"
work_dir="$root/runs/$run_id"
artifact_dir="$PWD/loongforge-release-artifacts"

if [[ -L "$artifact_dir" || (-e "$artifact_dir" && ! -d "$artifact_dir") ]]; then
  printf '%s\n' 'artifact: invalid directory' >&2
  exit 2
fi
rm -rf "$artifact_dir"
mkdir -p "$artifact_dir"

ssh_options=(-o BatchMode=yes -o ConnectTimeout=15)
ssh "${ssh_options[@]}" "$alias" "mkdir -p '$work_dir' '$remote_source'"
rsync --archive --delete --delete-excluded \
  --exclude='.git/' --exclude='.pytest_cache/' --exclude='__pycache__/' \
  "$source_dir/" "$alias:$remote_source/"

status=0
ssh "${ssh_options[@]}" "$alias" \
  "cd '$work_dir' && env -u DOCKER_HOST CI_CONFIG_PATH='$remote_config' \
   '$remote_source/.github/scripts/self_runner/run_release_regression.sh' \
   --image '$image' --source '$remote_source' --sha '$sha' \
   --llm-model '$llm_model' --vla-model '$vla_model'" || status=$?

if ! rsync --archive "$alias:$work_dir/loongforge-release-artifacts/" "$artifact_dir/"; then
  status=1
fi
exit "$status"
