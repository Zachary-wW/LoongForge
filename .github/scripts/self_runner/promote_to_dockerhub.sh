#!/usr/bin/env bash
# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

# Promote a validated image from iregistry to Docker Hub on the dedicated
# publisher host. `regctl` copies layers directly between registries and
# avoids the Docker daemon's slow local recompression path.

set -euo pipefail

internal_ref="${1:?internal image reference is required}"
dockerhub_ref="${2:?Docker Hub image reference is required}"
internal_host="${3:?internal registry host is required}"
dockerhub_host="${4:?Docker Hub registry host is required}"

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$script_dir/load_release_config.sh"
source "$script_dir/load_release_credentials.sh" iregistry
source "$script_dir/load_release_credentials.sh" dockerhub

regctl_bin="${LOONGFORGE_REGCTL_BIN:?LOONGFORGE_REGCTL_BIN is required}"

logged_in_hosts=()
cleanup_remote_logins() {
  local host
  for host in "${logged_in_hosts[@]}"; do
    printf -v remote_command \
      'HTTPS_PROXY=%q NO_PROXY=%q %q registry logout %q' \
      "$LOONGFORGE_DOCKERHUB_PROXY" \
      "$LOONGFORGE_DOCKERHUB_NO_PROXY" \
      "$regctl_bin" \
      "$host"
    ssh -o BatchMode=yes "$LOONGFORGE_RELEASE_PUBLISHER_ALIAS" \
      "$remote_command" >/dev/null 2>&1 || true
  done
}
trap cleanup_remote_logins EXIT INT TERM

login_remote_registry() {
  local host="$1"
  local username="$2"
  local password="$3"
  printf -v remote_command \
    'HTTPS_PROXY=%q NO_PROXY=%q %q registry login %q --user %q --pass-stdin' \
    "$LOONGFORGE_DOCKERHUB_PROXY" \
    "$LOONGFORGE_DOCKERHUB_NO_PROXY" \
    "$regctl_bin" \
    "$host" \
    "$username"
  printf '%s' "$password" |
    ssh -o BatchMode=yes "$LOONGFORGE_RELEASE_PUBLISHER_ALIAS" "$remote_command"
  logged_in_hosts+=("$host")
}

login_remote_registry "$internal_host" "$IREGISTRY_USERNAME" "$IREGISTRY_PASSWORD"
login_remote_registry "$dockerhub_host" "$DOCKERHUB_USERNAME" "$DOCKERHUB_TOKEN"

printf -v remote_command \
  'HTTPS_PROXY=%q NO_PROXY=%q %q image copy %q %q' \
  "$LOONGFORGE_DOCKERHUB_PROXY" \
  "$LOONGFORGE_DOCKERHUB_NO_PROXY" \
  "$regctl_bin" \
  "$internal_ref" \
  "$dockerhub_ref"
ssh -o BatchMode=yes "$LOONGFORGE_RELEASE_PUBLISHER_ALIAS" "$remote_command"
