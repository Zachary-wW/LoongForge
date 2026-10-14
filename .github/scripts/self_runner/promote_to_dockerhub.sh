#!/usr/bin/env bash
# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

# Pull, retag, and push a validated image on the dedicated publisher host.

set -euo pipefail
umask 077

internal_ref="${1:?internal image reference is required}"
dockerhub_ref="${2:?Docker Hub image reference is required}"
internal_host="${3:?internal registry host is required}"
dockerhub_host="${4:?Docker Hub registry host is required}"

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$script_dir/load_release_config.sh"
source "$script_dir/load_release_credentials.sh" iregistry
source "$script_dir/load_release_credentials.sh" dockerhub

redactor="$script_dir/../../../ci/redact_ci_artifact.py"
ssh_options=(-o BatchMode=yes -o ConnectTimeout=15 -o ServerAliveInterval=30 -o ServerAliveCountMax=3)
remote_config=""
destination_created=false
stage=prepare
source_id=""
push_digest=""
registry_digest=""
push_log="$(mktemp)"
mkdir -p loongforge-release-artifacts

remote() {
  ssh "${ssh_options[@]}" -n "$LOONGFORGE_RELEASE_PUBLISHER_ALIAS" "$@" \
    2> >(python3 "$redactor" --stream >&2)
}

docker_command() {
  printf 'HTTPS_PROXY=%q NO_PROXY=%q timeout --kill-after=30s %q env -u DOCKER_HOST -u DOCKER_CONTEXT DOCKER_CLI_EXPERIMENTAL=enabled docker --config %q' \
    "$LOONGFORGE_DOCKERHUB_PROXY" "$LOONGFORGE_DOCKERHUB_NO_PROXY" \
    "${LOONGFORGE_DOCKERHUB_STAGE_TIMEOUT_SECONDS:-3600}" "$remote_config"
  printf ' %q' "$@"
}

logged() {
  stage="$1"
  local command="$2" started=$SECONDS status
  printf 'dockerhub-promotion: %s started\n' "$stage"
  set +e
  if [[ "$stage" == login ]]; then
    ssh "${ssh_options[@]}" "$LOONGFORGE_RELEASE_PUBLISHER_ALIAS" "$command" 2>&1 |
      tee "$push_log" | python3 "$redactor" --stream
  else
    remote "$command" 2>&1 | tee "$push_log" | python3 "$redactor" --stream
  fi
  status="${PIPESTATUS[0]}"
  set -e
  printf 'dockerhub-promotion: %s finished in %ss (exit %s)\n' "$stage" "$((SECONDS - started))" "$status"
  return "$status"
}

cleanup() {
  local status=$? command
  trap - EXIT
  set +e
  if [[ "$remote_config" =~ ^/tmp/loongforge-publish[-.][A-Za-z0-9]+$ ]]; then
    for host in "$internal_host" "$dockerhub_host"; do
      remote "$(docker_command logout "$host")" >/dev/null 2>&1
    done
    if [[ "$destination_created" == true ]]; then
      remote "$(docker_command image rm "$dockerhub_ref")" >/dev/null 2>&1
    fi
    printf -v command 'find %q -mindepth 1 -depth -delete && rmdir %q' "$remote_config" "$remote_config"
    remote "$command" >/dev/null 2>&1
  fi
  python3 - "$status" "$stage" "$source_id" "$push_digest" "$registry_digest" <<'PY'
import json
import sys
from pathlib import Path
status, stage, source, pushed, registry = sys.argv[1:]
Path("loongforge-release-artifacts/dockerhub-promotion.result.json").write_text(
    json.dumps({"status": "passed" if status == "0" else "failed", "stage": stage,
                "exit_code": int(status), "source_image_id": source,
                "push_digest": pushed, "registry_digest": registry}) + "\n"
)
PY
  unlink "$push_log"
  exit "$status"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

remote_config="$(remote 'umask 077; mktemp -d /tmp/loongforge-publish.XXXXXXXX')"
[[ "$remote_config" =~ ^/tmp/loongforge-publish[-.][A-Za-z0-9]+$ ]] || {
  echo 'publisher: invalid temporary configuration' >&2
  exit 2
}
if remote "$(docker_command image inspect "$dockerhub_ref")" >/dev/null 2>&1; then
  echo 'publisher: destination tag already exists locally; refusing to replace it' >&2
  exit 2
fi

login_remote_registry() {
  stage=login
  local host="$1"
  local username="$2"
  local password="$3"
  printf '%s' "$password" | logged login "$(docker_command login --username "$username" --password-stdin "$host")"
}

login_remote_registry "$internal_host" "$IREGISTRY_USERNAME" "$IREGISTRY_PASSWORD"
login_remote_registry "$dockerhub_host" "$DOCKERHUB_USERNAME" "$DOCKERHUB_TOKEN"

logged pull "$(docker_command pull "$internal_ref")"
source_id="$(remote "$(docker_command image inspect --format '{{.Id}}' "$internal_ref")")"
[[ "$source_id" =~ ^sha256:[a-f0-9]{64}$ ]]
logged tag "$(docker_command tag "$internal_ref" "$dockerhub_ref")"
destination_created=true
logged push "$(docker_command push "$dockerhub_ref")"
push_digest="$(sed -nE 's/.*digest: (sha256:[a-f0-9]{64}).*/\1/p' "$push_log" | tail -1)"
[[ "$push_digest" =~ ^sha256:[a-f0-9]{64}$ ]]

stage=verify
verify_started=$SECONDS
printf 'dockerhub-promotion: verify started\n'
manifest="$(remote "$(docker_command manifest inspect --verbose "$dockerhub_ref")")"
registry_digest="$(printf '%s' "$manifest" | python3 -c '
import json, sys
data = json.load(sys.stdin)
entries = data if isinstance(data, list) else [data]
for entry in entries:
    manifest = entry.get("SchemaV2Manifest") or entry.get("OCIManifest") or {}
    if manifest.get("config", {}).get("digest") == sys.argv[1]:
        digest = entry.get("Descriptor", {}).get("digest", "")
        if digest == sys.argv[2]:
            print(digest)
            break
else:
    sys.exit("publisher: remote image content or digest does not match")
' "$source_id" "$push_digest")"
printf 'dockerhub-promotion: source image content and published digest verified\n'
printf 'dockerhub-promotion: verify finished in %ss\n' "$((SECONDS - verify_started))"
if [[ -n "${GITHUB_OUTPUT:-}" ]]; then
  printf 'verified=true\n' >> "$GITHUB_OUTPUT"
fi
