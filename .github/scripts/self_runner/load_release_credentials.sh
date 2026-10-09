#!/usr/bin/env bash
# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

# Usage: source load_release_credentials.sh iregistry|dockerhub

kind="${1:-}"
case "$kind" in
  iregistry)
    credentials="${LOONGFORGE_IREGISTRY_CREDENTIALS:-}"
    ;;
  dockerhub)
    credentials="${LOONGFORGE_DOCKERHUB_CREDENTIALS:-}"
    ;;
  *)
    printf '%s\n' 'release-credentials: invalid' >&2
    return 2
    ;;
esac

[[ -n "$credentials" && -f "$credentials" && ! -L "$credentials" ]] || {
  printf '%s\n' 'release-credentials: unavailable' >&2
  return 2
}

set -a
# shellcheck disable=SC1090
source "$credentials"
set +a

case "$kind" in
  iregistry)
    required=(IREGISTRY_USERNAME IREGISTRY_PASSWORD)
    ;;
  dockerhub)
    required=(DOCKERHUB_USERNAME DOCKERHUB_TOKEN)
    ;;
esac
for name in "${required[@]}"; do
  [[ -n "${!name:-}" ]] || {
    printf '%s\n' 'release-credentials: incomplete' >&2
    return 2
  }
done

mask_value() {
  [[ -n "${1:-}" ]] && printf '::add-mask::%s\n' "$1"
}

mask_value "${IREGISTRY_PASSWORD:-}"
mask_value "${IREGISTRY_USERNAME:-}"
mask_value "${DOCKERHUB_TOKEN:-}"
mask_value "${DOCKERHUB_USERNAME:-}"
