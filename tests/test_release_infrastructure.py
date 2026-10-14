# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import os
import subprocess
from pathlib import Path

import pytest

# --- tests/test_execution_slot.py ---
REPOSITORY_ROOT = Path(__file__).parents[1]
RESOLVE_SCRIPT = (
    REPOSITORY_ROOT
    / ".github"
    / "scripts"
    / "self_runner"
    / "resolve_execution_slot.sh"
)
REGRESSION_SCRIPT = (
    REPOSITORY_ROOT
    / ".github"
    / "scripts"
    / "self_runner"
    / "run_release_regression.sh"
)


def write_slot_config(tmp_path: Path) -> Path:
    config = tmp_path / "execution-slots.env"
    config.write_text(
        "\n".join(
            [
                "LOONGFORGE_EXECUTION_SLOT_PRIMARY_MODE=local",
                "LOONGFORGE_EXECUTION_SLOT_PRIMARY_DOCKER_HOST=",
                "LOONGFORGE_EXECUTION_SLOT_SECONDARY_MODE=ssh",
                "LOONGFORGE_EXECUTION_SLOT_SECONDARY_ALIAS=private-secondary",
                "LOONGFORGE_EXECUTION_SLOT_SECONDARY_DOCKER_HOST=ssh://private-secondary/private.sock",
                "LOONGFORGE_EXECUTION_SLOT_SECONDARY_ROOT=/private/root",
                "LOONGFORGE_EXECUTION_SLOT_SECONDARY_SOURCE=/private/source",
                "LOONGFORGE_EXECUTION_SLOT_SECONDARY_CONFIG=/private/config",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    config.chmod(0o600)
    return config


def test_secondary_slot_resolution_is_internal_and_complete(tmp_path):
    config = write_slot_config(tmp_path)
    github_env = tmp_path / "github.env"
    env = os.environ.copy()
    env.update(
        {
            "LOONGFORGE_EXECUTION_SLOT_CONFIG": str(config),
            "GITHUB_ENV": str(github_env),
        }
    )

    result = subprocess.run(
        ["bash", str(RESOLVE_SCRIPT), "secondary"],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert set(result.stdout.splitlines()) == {
        "::add-mask::private-secondary",
        "::add-mask::ssh://private-secondary/private.sock",
        "::add-mask::/private/root",
        "::add-mask::/private/source",
        "::add-mask::/private/config",
    }
    assert result.stderr == ""

    values = dict(
        line.split("=", 1)
        for line in github_env.read_text(encoding="utf-8").splitlines()
    )
    assert values["LOONGFORGE_EXECUTION_MODE"] == "ssh"
    assert values["LOONGFORGE_EXECUTION_ALIAS"] == "private-secondary"
    assert values["DOCKER_HOST"] == "ssh://private-secondary/private.sock"
    assert values["LOONGFORGE_EXECUTION_SOURCE"] == "/private/source"


def test_remote_slot_script_archives_evidence():
    script = REGRESSION_SCRIPT.read_text(encoding="utf-8")
    assert "LOONGFORGE_RELEASE_EVIDENCE_ROOT" in script
    assert "$evidence_root/$run_id/loongforge-release-artifacts" in script


def test_insecure_slot_configuration_fails_closed(tmp_path):
    config = write_slot_config(tmp_path)
    config.chmod(0o640)
    env = os.environ.copy()
    env["LOONGFORGE_EXECUTION_SLOT_CONFIG"] = str(config)

    result = subprocess.run(
        ["bash", str(RESOLVE_SCRIPT), "secondary"],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 2
    assert result.stdout == ""
    assert result.stderr.strip() == "execution-slot: insecure configuration"
    assert str(tmp_path) not in result.stderr

# --- tests/test_release_storage_check.py ---
STORAGE_SCRIPT = (
    Path(__file__).parents[1] / ".github" / "scripts" / "check_docker_storage.sh"
)


def test_storage_check_uses_remote_df_for_ssh_docker_host(tmp_path):
    fake_docker = tmp_path / "docker"
    fake_docker.write_text(
        "\n".join(
            [
                "#!/usr/bin/env bash",
                "if [[ \"$1\" == info ]]; then echo /remote/docker-data; exit 0; fi",
                "exit 2",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    fake_docker.chmod(0o755)
    fake_ssh = tmp_path / "ssh"
    fake_ssh.write_text(
        "\n".join(
            [
                "#!/usr/bin/env bash",
                "if [[ \"$1\" == -o ]]; then shift 2; fi",
                "case \"$2\" in \"df -Pk '/remote/docker-data'\"*) echo 262144000; exit 0;; esac",
                "exit 2",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    fake_ssh.chmod(0o755)
    env = os.environ.copy()
    env.update(
        {
            "DOCKER_BIN": str(fake_docker),
            "PATH": f"{tmp_path}:{env['PATH']}",
            "LOONGFORGE_EXECUTION_ALIAS": "private-secondary",
        }
    )
    result = subprocess.run(
        ["bash", str(STORAGE_SCRIPT), "250"],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert result.stdout == "docker-storage: ok\n"

# --- tests/test_release_promotion.py ---
PROMOTION_SCRIPT = (
    Path(__file__).parents[1]
    / ".github"
    / "scripts"
    / "self_runner"
    / "promote_to_dockerhub.sh"
)


def test_promotion_keeps_credentials_out_of_remote_commands(tmp_path):
    release_config = tmp_path / "release.env"
    release_config.write_text(
        "\n".join(
            [
                "DOCKER_HOST=ssh://publisher",
                "LOONGFORGE_RELEASE_PUBLISHER_ALIAS=publisher",
                "LOONGFORGE_DOCKERHUB_PROXY=http://proxy",
                "LOONGFORGE_DOCKERHUB_NO_PROXY=localhost",
                "LOONGFORGE_REGCTL_BIN=/remote/bin/regctl",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    release_config.chmod(0o600)

    internal_credentials = tmp_path / "internal.env"
    internal_credentials.write_text(
        "IREGISTRY_USERNAME=internal-user\nIREGISTRY_PASSWORD=internal-password\n",
        encoding="utf-8",
    )
    internal_credentials.chmod(0o600)
    hub_credentials = tmp_path / "dockerhub.env"
    hub_credentials.write_text(
        "DOCKERHUB_USERNAME=hub-user\nDOCKERHUB_TOKEN=hub-token\n",
        encoding="utf-8",
    )
    hub_credentials.chmod(0o600)

    command_log = tmp_path / "ssh-commands.log"
    stdin_log = tmp_path / "ssh-stdin.log"
    fake_ssh = tmp_path / "ssh"
    fake_ssh.write_text(
        "\n".join(
            [
                "#!/usr/bin/env bash",
                f"printf '\\n--- command ---\\n%s\\n' \"$*\" >> {command_log}",
                f"cat >> {stdin_log}",
                "printf '\\n' >> " + str(stdin_log),
                "exit 0",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    fake_ssh.chmod(0o755)

    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{tmp_path}:{env['PATH']}",
            "LOONGFORGE_RELEASE_CONFIG": str(release_config),
            "LOONGFORGE_IREGISTRY_CREDENTIALS": str(internal_credentials),
            "LOONGFORGE_DOCKERHUB_CREDENTIALS": str(hub_credentials),
        }
    )

    result = subprocess.run(
        [
            "bash",
            str(PROMOTION_SCRIPT),
            "registry.example/team/image:v1",
            "docker.io/loongforge/loongforge:v1",
            "registry.example",
            "docker.io",
        ],
        env=env,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert set(result.stdout.splitlines()) == {
        "::add-mask::internal-user",
        "::add-mask::internal-password",
        "::add-mask::hub-user",
        "::add-mask::hub-token",
    }
    assert result.stderr == ""
    commands = command_log.read_text(encoding="utf-8")
    stdin = stdin_log.read_text(encoding="utf-8")
    assert "registry login registry.example --user internal-user --pass-stdin" in commands
    assert "registry login docker.io --user hub-user --pass-stdin" in commands
    assert "image copy registry.example/team/image:v1 docker.io/loongforge/loongforge:v1" in commands
    assert "registry logout registry.example" in commands
    assert "registry logout docker.io" in commands
    assert "--host" not in commands
    assert "internal-password" not in commands
    assert "hub-token" not in commands
    assert "internal-password" in stdin
    assert "hub-token" in stdin

# --- tests/test_release_image_compatibility.py ---
MODULE_PATH = Path(__file__).parents[1] / ".github" / "scripts" / "image_policy.py"
SPEC = importlib.util.spec_from_file_location("image_policy", MODULE_PATH)
image_policy = importlib.util.module_from_spec(SPEC)
assert SPEC.loader
SPEC.loader.exec_module(image_policy)


def test_dockerfile_restores_diffusers_after_lerobot():
    dockerfile = (Path(__file__).parents[1] / "docker" / "Dockerfile").read_text(encoding="utf-8")

    assert "diffusers==0.38.0" in dockerfile
    assert "from diffusers import ConfigMixin, ModelMixin" in dockerfile
    assert dockerfile.index("install_lerobot") < dockerfile.index('install "diffusers==0.38.0"')


def test_image_policy_ignores_daemon_graph_driver_metadata():
    metadata = image_policy.image_metadata(
        '{"GraphDriver":{"Data":{"UpperDir":"/private/docker-data/diff"}},"Labels":{}}',
        "",
    )
    assert "/private/docker-data" not in metadata


def test_image_policy_accepts_inspect_array_and_ignores_graph_driver():
    metadata = image_policy.image_metadata(
        '[{"GraphDriver":{"Data":{"UpperDir":"/private/docker-data/diff"}},"Labels":{}}]',
        "",
    )
    assert "/private/docker-data" not in metadata

# --- tests/test_cleanup_debug_tags.py ---
CLEANUP_SCRIPT = Path(
    ".github/scripts/self_runner/cleanup_debug_tags.py",
)


def load_cleanup_module():
    spec = importlib.util.spec_from_file_location("cleanup_debug_tags", CLEANUP_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_split_image_reference_parses_registry_host_repository_and_tag():
    module = load_cleanup_module()
    host, repository, tag = module.split_image_reference(
        "registry.example.test/team/project:v0.2.3-ci_test",
    )
    assert host == "registry.example.test"
    assert repository == "team/project"
    assert tag == "v0.2.3-ci_test"


def test_split_image_reference_parses_dockerhub_reference():
    module = load_cleanup_module()
    host, repository, tag = module.split_image_reference(
        "docker.io/loongforge/loongforge:v0.2.3-ci_test",
    )
    assert host == "docker.io"
    assert module.normalize_dockerhub_repository(repository) == "loongforge/loongforge"
    assert tag == "v0.2.3-ci_test"


def test_split_image_reference_requires_a_tag():
    module = load_cleanup_module()
    with pytest.raises(ValueError):
        module.split_image_reference("registry.example.test/team/project")


def test_internal_registry_cleanup_treats_missing_tag_as_success(monkeypatch, capsys):
    module = load_cleanup_module()
    monkeypatch.setattr(
        module,
        "http_request",
        lambda *args, **kwargs: (404, {}, b""),
    )
    module.delete_internal_registry_tag(
        host="registry.example.test",
        image="registry.example.test/team/project:v0.2.3-ci_test",
        username="user",
        password="secret",
    )
    assert "internal registry: tag already absent" in capsys.readouterr().out

# --- tests/test_release_config.py ---
def run_loader(
    tmp_path: Path,
    *,
    credential_mode: int = 0o600,
    config_mode: int = 0o600,
    with_credentials: bool = True,
    config_complete: bool = True,
):
    internal = tmp_path / "internal-credentials"
    public = tmp_path / "public-credentials"
    internal.write_text("IREGISTRY_USERNAME=user\nIREGISTRY_PASSWORD=secret\n", encoding="utf-8")
    public.write_text("DOCKERHUB_USERNAME=user\nDOCKERHUB_TOKEN=token\n", encoding="utf-8")
    internal.chmod(credential_mode)
    public.chmod(credential_mode)
    config = tmp_path / "release-config"
    values = [
        "DOCKER_HOST=ssh://publisher",
        "LOONGFORGE_DOCKERHUB_PROXY=http://proxy",
        "LOONGFORGE_DOCKERHUB_NO_PROXY=localhost,.internal",
    ]
    if config_complete:
        values.append("LOONGFORGE_RELEASE_PUBLISHER_ALIAS=publisher")
    config.write_text("\n".join(values) + "\n", encoding="utf-8")
    config.chmod(config_mode)
    env = os.environ.copy()
    env["LOONGFORGE_RELEASE_CONFIG"] = str(config)
    if with_credentials:
        env["LOONGFORGE_IREGISTRY_CREDENTIALS"] = str(internal)
        env["LOONGFORGE_DOCKERHUB_CREDENTIALS"] = str(public)
    else:
        env.pop("LOONGFORGE_IREGISTRY_CREDENTIALS", None)
        env.pop("LOONGFORGE_DOCKERHUB_CREDENTIALS", None)
    return subprocess.run(
        ["bash", "-c", "source .github/scripts/self_runner/load_release_config.sh"],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def test_release_config_loader_accepts_private_complete_config(tmp_path):
    result = run_loader(tmp_path)
    assert result.returncode == 0
    assert result.stdout == ""
    assert result.stderr == ""


def test_release_config_loader_rejects_incomplete_config_without_paths(tmp_path):
    result = run_loader(tmp_path, with_credentials=False)
    assert result.returncode == 2
    assert result.stdout == ""
    assert result.stderr.strip() == "release-config: unavailable"
    assert str(tmp_path) not in result.stderr


def test_release_config_loader_rejects_config_missing_destination_fields(tmp_path):
    result = run_loader(tmp_path, config_complete=False)
    assert result.returncode == 2
    assert result.stdout == ""
    assert result.stderr.strip() == "release-config: incomplete"
    assert str(tmp_path) not in result.stderr


def test_release_config_loader_rejects_group_readable_credentials(tmp_path):
    result = run_loader(tmp_path, credential_mode=0o640)
    assert result.returncode == 2
    assert result.stdout == ""
    assert result.stderr.strip() == "release-config: insecure credentials"
    assert str(tmp_path) not in result.stderr


def test_release_config_loader_rejects_group_readable_config(tmp_path):
    result = run_loader(tmp_path, config_mode=0o640)
    assert result.returncode == 2
    assert result.stdout == ""
    assert result.stderr.strip() == "release-config: insecure configuration"
    assert str(tmp_path) not in result.stderr


def test_credentials_loader_masks_only_its_registry_under_set_e(tmp_path):
    hub_credentials = tmp_path / "hub-credentials"
    hub_credentials.write_text(
        "DOCKERHUB_USERNAME=hub-user\nDOCKERHUB_TOKEN=hub-token-value\n",
        encoding="utf-8",
    )
    hub_credentials.chmod(0o600)
    env = os.environ.copy()
    env["LOONGFORGE_DOCKERHUB_CREDENTIALS"] = str(hub_credentials)
    for name in ("DOCKERHUB_USERNAME", "DOCKERHUB_TOKEN", "IREGISTRY_PASSWORD"):
        env.pop(name, None)
    result = subprocess.run(
        [
            "bash", "-c",
            "set -euo pipefail; "
            "source .github/scripts/self_runner/load_release_credentials.sh dockerhub; "
            'test "$DOCKERHUB_TOKEN" = hub-token-value && echo loaded',
        ],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "loaded" in result.stdout
    assert "::add-mask::hub-token-value" in result.stdout
    assert "::add-mask::hub-user" in result.stdout
    assert "iregistry" not in result.stdout

# --- tests/test_runner_metadata_mask.py ---
def test_mask_script_masks_config_and_additional_values(tmp_path):
    image_config = tmp_path / "image-config"
    image_config.write_text("PRIVATE_PATH=/private/image/path\nEMPTY=\n", encoding="utf-8")
    release_config = tmp_path / "release-config"
    release_config.write_text("DOCKER_HOST=ssh://publisher\n", encoding="utf-8")
    execution_config = tmp_path / "execution-config"
    execution_config.write_text(
        "LOONGFORGE_EXECUTION_SLOT_SECONDARY_ALIAS=ssh-private-host\n",
        encoding="utf-8",
    )
    extra_values = tmp_path / "private-values"
    extra_values.write_text("# comment\nprivate.example.invalid\n", encoding="utf-8")
    hub_credentials = tmp_path / "hub-credentials"
    hub_credentials.write_text("DOCKERHUB_TOKEN=hub-secret-value\n", encoding="utf-8")
    env = os.environ.copy()
    env.update({
        "RUNNER_NAME": "private-runner-name",
        "INTERNAL_IMAGE_REPOSITORY": "registry.private.invalid/team/image",
        "DOCKERHUB_IMAGE": "registry.public.invalid/team/image",
        "CI_CONFIG_PATH_IMAGE": str(image_config),
        "LOONGFORGE_RELEASE_CONFIG": str(release_config),
        "LOONGFORGE_EXECUTION_SLOT_CONFIG": str(execution_config),
        "LOONGFORGE_DOCKERHUB_CREDENTIALS": str(hub_credentials),
        "LOONGFORGE_RUNNER_MASK_VALUES": str(extra_values),
    })
    result = subprocess.run(
        ["bash", ".github/scripts/self_runner/mask_runner_metadata.sh"],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    for value in (
        "private-runner-name",
        "registry.private.invalid/team/image",
        "registry.public.invalid/team/image",
        str(image_config),
        str(release_config),
        "/private/image/path",
        "ssh://publisher",
        "ssh-private-host",
        "private.example.invalid",
        "hub-secret-value",
        str(hub_credentials),
    ):
        assert f"::add-mask::{value}\n" in result.stdout
    assert "::add-mask::\n" not in result.stdout
