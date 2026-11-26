from __future__ import annotations

import sys
from pathlib import Path

import os

from core.stages.f1_intake.probe import resolve_trusted_interpreter, run_startup_probe

from helpers import create_sample_repo


def test_startup_probe_succeeds_in_isolated_dir_and_keeps_source_clean(
    tmp_path: Path,
) -> None:
    info = create_sample_repo(tmp_path)
    repo: Path = info["repo"]
    commit: str = info["head_commit"]
    probe_dir = tmp_path / "probe_out"

    result = run_startup_probe(
        repo_root=repo,
        expected_commit=commit,
        argv=[sys.executable, "scripts/train.py", "--config", "configs/pi05.yaml"],
        cwd=".",
        env={"SAFE_FLAG": "1", "OPENAI_API_KEY": "secret-should-be-redacted"},
        output_dir=probe_dir,
    )
    assert result.status == "passed"
    assert result.exit_code == 0
    assert result.observed_scope == ("startup_probe",)
    assert result.source_unchanged is True
    assert "OPENAI_API_KEY" not in result.sanitized_env
    assert result.sanitized_env["SAFE_FLAG"] == "1"
    assert (probe_dir / "probe_marker.json").exists()


def test_help_invocation_is_only_cli_parsed_never_startup_probe_passed(
    tmp_path: Path,
) -> None:
    info = create_sample_repo(tmp_path)
    repo: Path = info["repo"]
    commit: str = info["head_commit"]

    result = run_startup_probe(
        repo_root=repo,
        expected_commit=commit,
        argv=[sys.executable, "scripts/train.py", "--help"],
        cwd=".",
        env={},
        output_dir=tmp_path / "help_probe",
    )
    assert result.exit_code == 0
    assert result.status == "cli_parsed"
    assert result.observed_scope == ("cli_parsed",)
    assert "startup_probe" not in result.observed_scope


def test_startup_probe_detects_source_mutation_and_nonzero_exit(
    tmp_path: Path,
) -> None:
    info = create_sample_repo(tmp_path)
    repo: Path = info["repo"]
    commit: str = info["head_commit"]

    # Non-zero exit
    failed = run_startup_probe(
        repo_root=repo,
        expected_commit=commit,
        argv=[
            sys.executable,
            "scripts/train.py",
            "--config",
            "configs/pi05.yaml",
            "--mode",
            "fail",
        ],
        cwd=".",
        env={},
        output_dir=tmp_path / "fail_probe",
    )
    assert failed.status == "failed"
    assert failed.exit_code == 2
    assert "PROBE_EXIT_NONZERO" in failed.reason_codes

    # In default read-only materialized copy mode, write attempt is blocked at OS level
    # and original upstream repo stays completely untouched.
    ro_blocked = run_startup_probe(
        repo_root=repo,
        expected_commit=commit,
        argv=[
            sys.executable,
            "scripts/train.py",
            "--config",
            "configs/pi05.yaml",
            "--mutate-source",
        ],
        cwd=".",
        env={},
        output_dir=tmp_path / "ro_mut_probe",
        entrypoint_path="scripts/train.py",
        use_readonly_copy=True,
    )
    assert ro_blocked.status == "blocked"
    assert ro_blocked.source_unchanged is True
    assert "SOURCE_WRITE_ATTEMPT_BLOCKED" in ro_blocked.reason_codes
    assert not (repo / "MUTATED_BY_PROBE.txt").exists()

    # Even if run directly without read-only copy, pre/post snapshot catches mutation.
    mutated = run_startup_probe(
        repo_root=repo,
        expected_commit=commit,
        argv=[
            sys.executable,
            "scripts/train.py",
            "--config",
            "configs/pi05.yaml",
            "--mutate-source",
        ],
        cwd=".",
        env={},
        output_dir=tmp_path / "mut_probe",
        entrypoint_path="scripts/train.py",
        use_readonly_copy=False,
    )
    assert mutated.status == "blocked"
    assert mutated.source_unchanged is False
    assert "SOURCE_MUTATED_DURING_ANALYSIS" in mutated.reason_codes


def test_startup_probe_rejects_unrelated_argv_missing_marker_and_reserved_env(
    tmp_path: Path,
) -> None:
    info = create_sample_repo(tmp_path)
    repo: Path = info["repo"]
    commit: str = info["head_commit"]

    # 1) P0-1: `python -c pass` cannot pass startup_probe for `scripts/train.py`
    unrelated = run_startup_probe(
        repo_root=repo,
        expected_commit=commit,
        argv=[sys.executable, "-c", "pass"],
        cwd=".",
        env={},
        output_dir=tmp_path / "unrelated_probe",
        entrypoint_path="scripts/train.py",
    )
    assert unrelated.status == "blocked"
    assert "PROBE_ARGV_ENTRYPOINT_MISMATCH" in unrelated.reason_codes
    assert "startup_probe" not in unrelated.observed_scope

    # 2) P1-2: overriding reserved isolation env vars (HOME / HF_HUB_OFFLINE / TMPDIR) is rejected
    bad_env = run_startup_probe(
        repo_root=repo,
        expected_commit=commit,
        argv=[sys.executable, "scripts/train.py", "--config", "configs/pi05.yaml"],
        cwd=".",
        env={"HF_HUB_OFFLINE": "0", "HOME": "/tmp/evil"},
        output_dir=tmp_path / "bad_env_probe",
        entrypoint_path="scripts/train.py",
    )
    assert bad_env.status == "blocked"
    assert "RESERVED_ENV_OVERRIDE" in bad_env.reason_codes

    # 3) P0-1: script exiting 0 without structured probe_marker.json is rejected
    silent_repo_info = create_sample_repo(tmp_path / "silent_case")
    silent_repo: Path = silent_repo_info["repo"]
    (silent_repo / "scripts" / "silent.py").write_text("print('ok')\n", encoding="utf-8")
    from helpers import git

    git(silent_repo, "add", "scripts/silent.py")
    git(silent_repo, "commit", "-m", "add silent script")
    silent_commit = git(silent_repo, "rev-parse", "HEAD").stdout.strip()

    no_marker = run_startup_probe(
        repo_root=silent_repo,
        expected_commit=silent_commit,
        argv=[sys.executable, "scripts/silent.py"],
        cwd=".",
        env={},
        output_dir=tmp_path / "no_marker_probe",
        entrypoint_path="scripts/silent.py",
    )
    assert no_marker.exit_code == 0
    assert no_marker.status == "blocked"
    assert "PROBE_MARKER_MISSING" in no_marker.reason_codes
    assert "startup_probe" not in no_marker.observed_scope

    # 4) Batch 2 P0: arbitrary external shell wrapper writing a fake marker is rejected!
    wrapper = tmp_path / "fake_marker_probe.sh"
    wrapper.write_text(
        "#!/bin/sh\n"
        'echo \'{"event":"startup_probe_ok"}\' > "$INTAKE_PROBE_OUTPUT_DIR/probe_marker.json"\n'
        "exit 0\n",
        encoding="utf-8",
    )
    wrapper.chmod(0o755)
    wrapped = run_startup_probe(
        repo_root=repo,
        expected_commit=commit,
        argv=[str(wrapper), "scripts/train.py"],
        cwd=".",
        env={},
        output_dir=tmp_path / "wrapper_probe",
        entrypoint_path="scripts/train.py",
    )
    assert wrapped.status == "blocked"
    assert "UNTRUSTED_PROBE_LAUNCHER" in wrapped.reason_codes
    assert "startup_probe" not in wrapped.observed_scope

    # 5) Even a fake wrapper named `python3` (shebang text script) is rejected!
    fake_py = tmp_path / "python3"
    fake_py.write_text(
        "#!/bin/sh\n"
        'echo \'{"event":"startup_probe_ok"}\' > "$INTAKE_PROBE_OUTPUT_DIR/probe_marker.json"\n'
        "exit 0\n",
        encoding="utf-8",
    )
    fake_py.chmod(0o755)
    wrapped_py = run_startup_probe(
        repo_root=repo,
        expected_commit=commit,
        argv=[str(fake_py), "scripts/train.py"],
        cwd=".",
        env={},
        output_dir=tmp_path / "fake_py_probe",
        entrypoint_path="scripts/train.py",
    )
    assert wrapped_py.status == "blocked"
    assert "UNTRUSTED_PROBE_LAUNCHER" in wrapped_py.reason_codes

    # 6) Batch 2 final P0: task setting env.PATH to a dir with malicious `python3` wrapper
    # and argv=["python3", "scripts/train.py"] MUST be rejected (cannot hijack bare `python3`).
    malicious_bin = tmp_path / "malicious_bin"
    malicious_bin.mkdir()
    malicious_py = malicious_bin / "python3"
    malicious_py.write_text(
        "#!/bin/sh\n"
        'cat <<EOF > "$INTAKE_PROBE_OUTPUT_DIR/probe_marker.json"\n'
        '{"event":"startup_probe_ok","source_commit":"$INTAKE_PROBE_COMMIT",'
        '"entrypoint_path":"$INTAKE_PROBE_ENTRYPOINT",'
        '"argv_sha256":"$INTAKE_PROBE_ARGV_SHA256","probe_nonce":"$INTAKE_PROBE_NONCE"}\n'
        "EOF\n"
        "exit 0\n",
        encoding="utf-8",
    )
    malicious_py.chmod(0o755)
    path_hijack = run_startup_probe(
        repo_root=repo,
        expected_commit=commit,
        argv=["python3", "scripts/train.py", "--config", "configs/pi05.yaml"],
        cwd=".",
        env={"PATH": f"{malicious_bin}:/usr/bin:/bin"},
        output_dir=tmp_path / "path_hijack_probe",
        entrypoint_path="scripts/train.py",
    )
    assert path_hijack.status == "blocked"
    assert "RESERVED_ENV_OVERRIDE" in path_hijack.reason_codes
    assert "startup_probe" not in path_hijack.observed_scope

    # Also verify PYTHONPATH / PYTHONHOME / PYTHONSTARTUP overrides are rejected.
    pypath_hijack = run_startup_probe(
        repo_root=repo,
        expected_commit=commit,
        argv=["python3", "scripts/train.py", "--config", "configs/pi05.yaml"],
        cwd=".",
        env={"PYTHONPATH": str(malicious_bin)},
        output_dir=tmp_path / "pypath_hijack_probe",
        entrypoint_path="scripts/train.py",
    )
    assert pypath_hijack.status == "blocked"
    assert "RESERVED_ENV_OVERRIDE" in pypath_hijack.reason_codes

    # 7) Batch 2 final P1: any explicit `python3` path that does NOT resolve to
    # `os.path.realpath(sys.executable)` (including system prefixes like `/usr/bin/python3`,
    # `/opt/homebrew/bin/python3`, or a non-shebang binary named `python3`) MUST be rejected!
    current_real = os.path.realpath(sys.executable)
    assert resolve_trusted_interpreter("python3") == current_real
    assert resolve_trusted_interpreter(sys.executable) == current_real

    # Non-shebang binary file named `python3` in an arbitrary directory
    fake_elf = tmp_path / "bin_other" / "python3"
    fake_elf.parent.mkdir()
    fake_elf.write_bytes(b"\xcf\xfa\xed\xfe" + b"\x00" * 64)
    fake_elf.chmod(0o755)
    assert resolve_trusted_interpreter(str(fake_elf)) is None

    bin_probe = run_startup_probe(
        repo_root=repo,
        expected_commit=commit,
        argv=[str(fake_elf), "scripts/train.py", "--config", "configs/pi05.yaml"],
        cwd=".",
        env={},
        output_dir=tmp_path / "fake_elf_probe",
        entrypoint_path="scripts/train.py",
    )
    assert bin_probe.status == "blocked"
    assert "UNTRUSTED_PROBE_LAUNCHER" in bin_probe.reason_codes

    # System prefix paths (`/usr/bin/python3`, `/opt/homebrew/bin/python3`, `/usr/local/bin/python3`)
    # are rejected whenever they do not resolve to `os.path.realpath(sys.executable)`.
    for candidate in (
        "/usr/bin/python3",
        "/usr/local/bin/python3",
        "/opt/homebrew/bin/python3",
    ):
        if os.path.realpath(candidate) != current_real:
            assert resolve_trusted_interpreter(candidate) is None
            sys_prefix_probe = run_startup_probe(
                repo_root=repo,
                expected_commit=commit,
                argv=[candidate, "scripts/train.py", "--config", "configs/pi05.yaml"],
                cwd=".",
                env={},
                output_dir=tmp_path / f"prefix_{Path(candidate).parent.name}",
                entrypoint_path="scripts/train.py",
            )
            assert sys_prefix_probe.status == "blocked"
            assert "UNTRUSTED_PROBE_LAUNCHER" in sys_prefix_probe.reason_codes
