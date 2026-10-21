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

redactor="$script_dir/../redact_ci_artifact.py"
ssh_options=(-o BatchMode=yes -o ConnectTimeout=15 -o ServerAliveInterval=30 -o ServerAliveCountMax=3)
remote_config=""
destination_created=false
stage=prepare
login_target=""
source_id=""
push_digest=""
registry_digest=""
push_log="$(mktemp)"
mkdir -p loongforge-release-artifacts
promotion_log=loongforge-release-artifacts/dockerhub-promotion.log
: > "$promotion_log"

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
  printf 'dockerhub-promotion: %s started\n' "$stage" | tee -a "$promotion_log"
  set +e
  if [[ "$stage" == login_* ]]; then
    ssh "${ssh_options[@]}" "$LOONGFORGE_RELEASE_PUBLISHER_ALIAS" "$command" 2>&1 |
      tee "$push_log" | python3 "$redactor" --stream | tee -a "$promotion_log"
  else
    remote "$command" 2>&1 | tee "$push_log" | python3 "$redactor" --stream | tee -a "$promotion_log"
  fi
  status="${PIPESTATUS[0]}"
  set -e
  printf 'dockerhub-promotion: %s finished in %ss (exit %s)\n' \
    "$stage" "$((SECONDS - started))" "$status" | tee -a "$promotion_log"
  return "$status"
}

cleanup() {
  local status=$? command archive
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
  python3 - "$status" "$stage" "$login_target" "$source_id" "$push_digest" "$registry_digest" <<'PY'
import json
import sys
from pathlib import Path
status, stage, login_target, source, pushed, registry = sys.argv[1:]
Path("loongforge-release-artifacts/dockerhub-promotion.result.json").write_text(
    json.dumps({"status": "passed" if status == "0" else "failed", "stage": stage,
                "login_target": login_target, "exit_code": int(status), "source_image_id": source,
                "push_digest": pushed, "registry_digest": registry}) + "\n"
)
PY
  if [[ -n "${LOONGFORGE_RELEASE_EVIDENCE_ROOT:-}" && "${GITHUB_RUN_ID:-}" =~ ^[0-9]+$ ]]; then
    archive="$LOONGFORGE_RELEASE_EVIDENCE_ROOT/$GITHUB_RUN_ID/loongforge-release-artifacts"
    if ! { mkdir -p "$archive" && cp "$promotion_log" \
        loongforge-release-artifacts/dockerhub-promotion.result.json "$archive/"; } >/dev/null 2>&1; then
      printf '%s\n' 'publisher: evidence archive unavailable' >&2
    fi
  fi
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
  login_target="$1"
  stage="login_$login_target"
  local host="$2" username="$3" password="$4" attempt status
  for attempt in 1 2 3; do
    if printf '%s' "$password" | logged "$stage" \
        "$(docker_command login --username "$username" --password-stdin "$host")"; then
      login_target=""
      return 0
    else
      status=$?
    fi
    if [[ "$attempt" == 3 ]] || ! grep -Eqi \
        'timed out|timeout|deadline exceeded|request cancel[l]?ed|connection reset|connection refused|temporary failure|unexpected EOF|service unavailable|bad gateway' \
        "$push_log"; then
      return "$status"
    fi
    printf 'dockerhub-promotion: %s temporary network failure; retrying (%s/3)\n' \
      "$stage" "$((attempt + 1))" | tee -a "$promotion_log"
    sleep "$((attempt * 2))"
  done
}

login_remote_registry iregistry "$internal_host" "$IREGISTRY_USERNAME" "$IREGISTRY_PASSWORD"
login_remote_registry dockerhub "$dockerhub_host" "$DOCKERHUB_USERNAME" "$DOCKERHUB_TOKEN"

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
printf 'dockerhub-promotion: verify started\n' | tee -a "$promotion_log"
# Read the tag through the authenticated Registry API. Older Docker clients can
# fail manifest inspection even after their daemon successfully pushes an image.
# Credentials stay in the private configuration created by this promotion.
printf -v verify_command \
  'HTTPS_PROXY=%q NO_PROXY=%q timeout --kill-after=10s 180s python3 - %q %q %q %q' \
  "$LOONGFORGE_DOCKERHUB_PROXY" "$LOONGFORGE_DOCKERHUB_NO_PROXY" \
  "$remote_config" "$dockerhub_ref" "$source_id" "$push_digest"
registry_digest="$(ssh "${ssh_options[@]}" "$LOONGFORGE_RELEASE_PUBLISHER_ALIAS" "$verify_command" \
  2> >(python3 "$redactor" --stream | tee -a "$promotion_log" >&2) <<'VERIFY_PY'
import hashlib
import http.client
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path


def registry_request(request, phase):
    for attempt in range(4):
        try:
            with urllib.request.urlopen(request, timeout=15) as response:
                body = response.read(4 * 1024 * 1024 + 1)
                if len(body) > 4 * 1024 * 1024:
                    raise RuntimeError("publisher: verify response is too large")
                return response.headers, body
        except urllib.error.HTTPError as error:
            status = error.code
            error.close()
            reason = f"HTTP {status}"
            retry = status in (408, 429) or 500 <= status < 600 or (
                status == 404 and phase == "manifest"
            )
        except (urllib.error.URLError, TimeoutError, ConnectionError, http.client.IncompleteRead):
            reason, retry = "network error", True
        # Do not log response bodies, URLs or authorization headers.
        print(
            f"publisher: verify {phase} attempt {attempt + 1}/4: {reason}",
            file=sys.stderr,
        )
        if not retry or attempt == 3:
            raise RuntimeError(f"publisher: verify {phase} failed: {reason}")
        time.sleep(2 ** (attempt + 1))


def decode_response(body, phase):
    try:
        payload = json.loads(body)
    except (ValueError, UnicodeError):
        raise RuntimeError(f"publisher: verify {phase} returned invalid JSON") from None
    if not isinstance(payload, dict):
        raise RuntimeError(f"publisher: verify {phase} returned invalid JSON")
    return payload


def verify(config_dir, image, source_id, push_digest):
    repository, tag = image.rsplit(":", 1)
    host, separator, remainder = repository.partition("/")
    hub_hosts = ("docker.io", "index.docker.io", "registry-1.docker.io")
    if separator and ("." in host or ":" in host or host == "localhost"):
        if host not in hub_hosts:
            raise RuntimeError("publisher: verify requires a Docker Hub repository")
        repository = remainder
    if "/" not in repository:
        repository = "library/" + repository

    config = json.loads((Path(config_dir) / "config.json").read_text())
    auths = config.get("auths", {})
    auth = next(
        (
            auths[key].get("auth")
            for key in ("https://index.docker.io/v1/", *hub_hosts)
            if auths.get(key, {}).get("auth")
        ),
        "",
    )
    if not auth:
        raise RuntimeError("publisher: verify requires the promotion's Docker Hub login")
    query = urllib.parse.urlencode(
        {"service": "registry.docker.io", "scope": f"repository:{repository}:pull"}
    )
    _, body = registry_request(
        urllib.request.Request(
            "https://auth.docker.io/token?" + query,
            headers={"Authorization": "Basic " + auth},
        ),
        "authentication",
    )
    payload = decode_response(body, "authentication")
    token = payload.get("token") or payload.get("access_token")
    if not isinstance(token, str) or not token:
        raise RuntimeError("publisher: verify authentication returned no token")

    path = urllib.parse.quote(repository, safe="/") + "/manifests/" + urllib.parse.quote(tag, safe="")
    headers, body = registry_request(
        urllib.request.Request(
            "https://registry-1.docker.io/v2/" + path,
            headers={
                "Authorization": "Bearer " + token,
                "Accept": (
                    "application/vnd.docker.distribution.manifest.v2+json, "
                    "application/vnd.oci.image.manifest.v1+json"
                ),
            },
        ),
        "manifest",
    )
    # Hash the manifest bytes as well as checking the registry's digest header.
    digest = "sha256:" + hashlib.sha256(body).hexdigest()
    declared_digest = headers.get("Docker-Content-Digest")
    if digest != push_digest or (declared_digest and declared_digest != push_digest):
        raise RuntimeError("publisher: remote tag does not match the pushed digest")
    manifest = decode_response(body, "manifest")
    config = manifest.get("config")
    if not isinstance(config, dict) or config.get("digest") != source_id:
        raise RuntimeError("publisher: remote image content does not match the source")
    return digest


if __name__ == "__main__":
    try:
        print(verify(*sys.argv[1:]))
    except RuntimeError as error:
        sys.exit(str(error))
    except (OSError, ValueError, TypeError, KeyError):
        sys.exit("publisher: verify received invalid configuration or response")
VERIFY_PY
)"
[[ "$registry_digest" == "$push_digest" ]]
printf 'dockerhub-promotion: source image content and published digest verified\n' | tee -a "$promotion_log"
printf 'dockerhub-promotion: verify finished in %ss\n' "$((SECONDS - verify_started))" | tee -a "$promotion_log"
if [[ -n "${GITHUB_OUTPUT:-}" ]]; then
  printf 'verified=true\n' >> "$GITHUB_OUTPUT"
fi
