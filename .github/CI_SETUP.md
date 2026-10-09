# CI repository setup

Repository files define the workflows, but the following settings must be
created by an organization or repository administrator.

## Repository Variables

Configure the following Repository Variables in GitHub Settings. Runner values
are JSON arrays, not comma-separated strings. Set the custom labels to the
labels actually registered on the target runners; the `a`/`p` labels in
runner-local configuration are canonical aliases,
not assumptions made by the workflows. `llm_vlm` resolves to `CI_RUNNER_A` and
`embodied` resolves to `CI_RUNNER_P`; each suite runner builds its own local
candidate image when `--build-image` is requested.

- `CI_RUNNER_A`: JSON label array for the A-card regression runner.
- `CI_RUNNER_P`: JSON label array for the P-card regression runner.
- `CI_REVIEW_RUNNER`: JSON label array for the Claude review runner.
- `CI_RELEASE_RUNNER`: JSON label array for a GPU Docker runner with at least
  250 GiB free on Docker's storage filesystem. Its GPU target determines the
  release image architecture in the same way as a candidate build.
- `CI_ENABLE_LLM_VLM`: `true` only when the A-card suite is provisioned;
  otherwise `llm_vlm` dispatch is intentionally rejected.

Configure these Claude review repository variables:

- `CLAUDE_REVIEW_MODEL` is required and must be a model ID accepted by the
  configured `ANTHROPIC_BASE_URL`.
- `CLAUDE_CODE_EXECUTABLE` may point to a pinned Claude Code binary already
  installed on every selected review runner. When set, the action skips its
  network installation step. Leave it empty to use the action's automatic
  installation.

When a review fails, the workflow prints only SDK result metadata
(`api_error_status`, `terminal_reason`, turn count, and timing). The full
`claude-execution-output.json` contains model/tool content and remains in the
runner temp directory only; do not upload it or print it to public logs.

Each self-hosted runner must provide `CI_CONFIG_PATH_IMAGE` in its service
environment, pointing at that machine's private `image.env`. The service
environment and that file together must provide the values shown in
`.github/ci-config.example.env`, including the default image and isolated
mounts and BuildKit mirror paths. Do not put registry credentials or signed
source URLs in the repository; keep them in the operator-managed runner
configuration only. The candidate-image builder passes proxy settings without
embedded credentials and uses the configured APT, PyPI, and source-manifest
files as ephemeral BuildKit secrets.

The protected release runner must also provide `LOONGFORGE_RELEASE_CONFIG`
and `LOONGFORGE_RUNNER_MASK_SCRIPT`; `LOONGFORGE_RUNNER_MASK_VALUES` may point
to an additional newline-delimited private-literal list. The mask script path
points to an operator-controlled copy of `mask_runner_metadata.sh` outside the checkout.
It runs before checkout or any third-party Action and registers the runner
name, hostname, private config paths and values, proxy settings, and workspace
paths with GitHub's log masker. Release Docker command output passes through
the repository redactor as a second layer; raw Docker output must not be added
to this workflow.

Each suite runner must also provide a working Docker Buildx plugin. The PR
checkout is the build context and the builder selects an operator-managed
trusted Dockerfile; configured BuildKit secrets are mounted for its
installation steps. An `IMAGE_DOCKERFILE` override must also be
operator-managed. Verify Buildx with `docker buildx version`;
installing the CLI plugin does not require restarting the Docker daemon.
The release job builds with the daemon's own BuildKit through
`DOCKER_BUILDKIT=1 docker build`; it deliberately does not create a
`docker-container` builder, because bootstrapping one pulls a builder image
from Docker Hub and an internal-only runner cannot reach it.

The builder forwards the runner's proxy settings into the image build, so the
runner's `NO_PROXY` must list every internal APT and PyPI mirror host. A mirror
that is only reachable directly fails with a gateway error when the build
sends it through the proxy.
The release runner uses this same builder and runner-local image config. Set
`LOONGFORGE_ALLOW_RELEASE_IMAGE_BUILD=true` only on the protected runner named
by `CI_RELEASE_RUNNER`; keep it false on general-purpose runners. Set
`LOONGFORGE_RELEASE_IMAGE_TAG_PREFIX` to a string that describes the actual
operator-managed `IMAGE_BASE_IMAGE`. The current protected base image is
Ubuntu 24.04, CUDA 13.3, PyTorch 2.13.0a0, and Python 3.12, so its prefix is
`ubuntu24.04-cu13.3-torch2.13.0a0-py312`.

Set `LOONGFORGE_RELEASE_RUN_REGRESSION=false` while the GPU runner is occupied;
set it to `true` before enabling the two model regressions.

`LOONGFORGE_RELEASE_TAG_SUFFIX` is optional and empty by default. Set it, for
example to `-citest`, on a non-production runner so a rehearsal appends the
marker to both the internal image tag and the Docker Hub tag instead of
claiming the real release tags.

Candidate builds require 250 GiB free by default on the filesystem containing
Docker's `DockerRootDir`. Set `LOONGFORGE_MIN_DOCKER_FREE_GB` to another integer
in the runner config, or set it explicitly to an empty value to disable the
check. `LOONGFORGE_MIN_FREE_GPU_MB` follows the same unset/default and
empty/disabled convention. GPU query errors fail closed and logs do not expose
device indices, counts, memory sizes, or runner paths.

Remove obsolete `INTERNAL_CI_*` variables after the old internal CI workflow
has been retired. They are not consumed by the current workflows and may
expose runner-local paths through the public repository configuration.

Create protected `pypi-release`, `iregistry-release`, and `dockerhub-release`
Environments for the release workflow. They are approval gates; registry
credentials remain on the protected release runner and are not stored in
GitHub. Releases push the validated internal image first, then the same
release runner drives the publisher Docker daemon over SSH and pushes only
`<DOCKERHUB_IMAGE>:<version>`; the `latest` tag is never moved. A manual
`workflow_dispatch` builds, scans, and regresses the image but does not push to
either registry.

PyPI publishing is opt-in and off by default. Set the repository variable
`PUBLISH_TO_PYPI` to `true` only after a pending or existing PyPI Trusted
Publisher is registered for owner `baidu-baige`, repository `LoongForge`,
workflow `release.yml`, and environment `pypi-release`. While it is unset or
false, a version tag builds, regresses, and promotes the image without a PyPI
step, and a failed or missing PyPI configuration cannot block a release.

Both candidate and release images pass the trusted image policy before use or
push. It rejects `bcecmd`, common cloud credential files, persisted source
manifests, AK/SK or signed-authorization metadata, and internal endpoints or
runner paths. The filesystem text scan excludes the copied
`/workspace/LoongForge` source tree because that tree is already covered by the
repository sensitive scan and contains historical documentation fixtures. It
also excludes Docker's runtime-generated host, hostname, and resolver files;
file-name checks for prohibited executables and credential files still cover
the whole image.

The submodule sync workflow uses a GitHub App token so it can push through the
repository's protected branch rules. Configure `SUBMODULE_SYNC_APP_ID` and
`SUBMODULE_SYNC_APP_PRIVATE_KEY` as repository secrets, and install that App on
this repository with permission to write contents.

## Runner labels

- A-card regression runner: `self-hosted`, plus the registered A-card custom
  label (the example alias is `a`).
- P-card regression runner: `self-hosted`, plus the registered P-card custom
  label (the example alias is `p`).
- Candidate images are built on the same suite runner that performs regression.
  The trusted image wrapper detects the runner target from `nvidia-smi`: compute
  capability 8.x selects A/Ampere, while 10.x, 11.x, or 12.x selects
  P/Blackwell. Mixed or unknown GPU architectures fail closed.

## First activation

Enable the `ok-to-test`, `gpu-regression`, `gpu-invalidate`, and
`gpu-watchdog` workflows on the default branch, then verify a
maintainer-dispatched run on each labeled suite
runner. Confirm that the PR `gpu-regression` check progresses through queued, optional
candidate build, regression, and final status. Push a second commit while a GPU
run is active and confirm that the old check becomes cancelled and the runner's
targeted cleanup removes its regression container and candidate image where
possible. Cancellation cleanup is best-effort and must not globally prune the
shared BuildKit cache.

Configure branch protection to require the PR job named `static-checks`. A manual
dispatch ends in `manual-static-checks` and is intended only for diagnostics; it must
not satisfy the PR requirement.

Create the `baidu-baige/loongforge-maintainers` Team before activating the
`.github/CODEOWNERS` rule. The repository should use a `master-protection`
Ruleset requiring one non-author Code Owner approval, resolved conversations,
an up-to-date branch, and the `static-checks` check. Use a separate
`release-tag-protection` Ruleset for immutable `v*` tags.

Keep the default `GITHUB_TOKEN` permission read-only and require approval for
first-time fork workflows. Because this repository uses self-hosted runners,
only trusted dispatch workflows should reach those runners. The current
repository allows all third-party Actions and does not require SHA pinning;
enable mandatory pinning only after every `@vN` reference has been replaced by
an audited full commit SHA.

Set these optional repository variables when the defaults do not match the
organization layout:

- `INTERNAL_IMAGE_REPOSITORY` is required for a release tag: the full image
  repository without a tag. The login host is derived from it.
- `DOCKERHUB_IMAGE`, default `docker.io/loongforge/loongforge`.
- `PUBLISH_TO_PYPI`, default unset: `true` enables the optional PyPI release
  step for version tags.

The release workflow builds one image and creates one container on
`CI_RELEASE_RUNNER`. It runs `deepseek_v2_lite` and then `pi05_ddp` in that
same container when `LOONGFORGE_RELEASE_RUN_REGRESSION=true`. While that
runner-local switch is false, image build and registry promotion continue
without the GPU regression step.

The release runner's operator-managed files must define:

The release runner's operator-managed release directory contains
one non-secret configuration file and one credential file for each registry.

The non-secret configuration file contains an SSH Docker client destination
and its matching SSH alias. The credential files hold the push-capable
internal registry account and the Docker Hub login name plus Docker Hub PAT,
and the runner service environment points at both of them.
Image repository names stay in the GitHub repository variables listed above.
All three files must be runner-local, mode `0600`, and excluded from version
control. The workflow uses a per-job Docker config directory and removes it
during cleanup.

The release runner needs Docker, read access to the internal registry, and
write access to Docker Hub through the SSH-selected daemon. The publishing
host needs at least 40 GiB free on the filesystem holding its
`DockerRootDir`, but it does not need a GitHub runner or a checkout.
