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

ssh -o BatchMode=yes "$LOONGFORGE_RELEASE_PUBLISHER_ALIAS" \
  "HTTPS_PROXY='$LOONGFORGE_DOCKERHUB_PROXY' \
   NO_PROXY='$LOONGFORGE_DOCKERHUB_NO_PROXY' \
   '$regctl_bin' image copy \
   --host 'reg=$internal_host,user=$IREGISTRY_USERNAME,pass=$IREGISTRY_PASSWORD' \
   --host 'reg=$dockerhub_host,user=$DOCKERHUB_USERNAME,pass=$DOCKERHUB_TOKEN' \
   '$internal_ref' '$dockerhub_ref'"
