# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

import subprocess
import os
import shlex
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).parent.parent
RELEASE_REGRESSION_SCRIPT = (
    REPOSITORY_ROOT
    / ".github"
    / "scripts"
    / "self_runner"
    / "run_release_regression.sh"
)
CREATE_CONTAINER_SCRIPT = (
    REPOSITORY_ROOT
    / ".github"
    / "scripts"
    / "self_runner"
    / "create_container.sh"
)


def test_release_regression_script_is_valid_shell():
    result = subprocess.run(
        ["bash", "-n", str(RELEASE_REGRESSION_SCRIPT)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert result.stdout == ""
    assert result.stderr == ""


def test_release_regression_sets_single_node_distributed_environment():
    script = RELEASE_REGRESSION_SCRIPT.read_text(encoding="utf-8")

    for name, value in (
        ("RANK", "0"),
        ("WORLD_SIZE", "1"),
        ("MASTER_ADDR", "127.0.0.1"),
    ):
        assert f'-e "{name}={value}"' in script

    assert '-e "MASTER_PORT=$master_port"' in script
    assert "master_port=$((20000 + ($$ % 30000)))" in script


def test_release_container_copies_megatron_into_writable_source(tmp_path):
    source = tmp_path / "source"
    megatron = source / "third_party" / "Loong-Megatron"
    megatron_marker = megatron / "megatron" / "core" / "transformer"
    megatron_marker.mkdir(parents=True)
    (megatron_marker / "hyper_connection.py").write_text("stub\n", encoding="utf-8")
    (source / "loongforge.txt").write_text("stub\n", encoding="utf-8")

    host_data = tmp_path / "host-data"
    host_llm_data = tmp_path / "host-llm-data"
    host_output = tmp_path / "host-output"
    container_source = tmp_path / "container-source"
    for directory in (host_data, host_llm_data, host_output):
        directory.mkdir()

    docker_log = tmp_path / "docker.log"
    fake_docker = tmp_path / "fake-docker"
    fake_docker.write_text(
        "\n".join(
            [
                "#!/usr/bin/env bash",
                "set -e",
                'if [[ "$1" == "container" && "$2" == "inspect" ]]; then exit 1; fi',
                'if [[ "$1" == "run" ]]; then',
                f'    printf \'%s\\n\' "$*" >> {shlex.quote(str(docker_log))}',
                "    exit 0",
                "fi",
                'if [[ "$1" == "exec" ]]; then',
                "    shift 2",
                '    exec "$@"',
                "fi",
                "exit 2",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    fake_docker.chmod(0o755)

    env = os.environ.copy()
    env.update(
        {
            "DOCKER_BIN": str(fake_docker),
            "LOONGFORGE_PULL_IMAGE": "false",
            "LOONGFORGE_HOST_DATA_ROOT": str(host_data),
            "LOONGFORGE_CONTAINER_DATA_ROOT": "/ci-data",
            "LOONGFORGE_HOST_LLM_DATA_ROOT": str(host_llm_data),
            "LOONGFORGE_CONTAINER_LLM_DATA_ROOT": "/ci-llm-data",
            "LOONGFORGE_HOST_OUTPUT_ROOT": str(host_output),
            "LOONGFORGE_CONTAINER_OUTPUT_ROOT": "/ci-output",
            "LOONGFORGE_CONTAINER_SOURCE": str(container_source),
            "LOONGFORGE_CONTAINER_SOURCE_MOUNT": str(source),
            "TRITON_LIBCUDA_PATH": "/opt/triton/libcuda",
            "LOONGFORGE_GPU_DEVICE": "/dev/null",
        }
    )

    result = subprocess.run(
        [
            str(CREATE_CONTAINER_SCRIPT),
            "registry.example/loongforge:test",
            str(source),
            "release",
            "loongforge-test-container",
        ],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert result.stdout == ""
    assert result.stderr == ""
    assert (container_source / "loongforge.txt").is_file()
    assert (
        container_source
        / "third_party"
        / "Loong-Megatron"
        / "megatron"
        / "core"
        / "transformer"
        / "hyper_connection.py"
    ).is_file()

    docker_args = docker_log.read_text(encoding="utf-8")
    assert "--device=/dev/null" in docker_args
    assert f"{source}:{source}:ro" in docker_args
    assert "/workspace/Loong-Megatron:ro" not in docker_args
