import hashlib
from pathlib import Path

import pytest

from core.artifacts import ArtifactError, PackageRef, publish, verify


def _stage_sample_package(
    root: Path,
    name: str = "f1-release-01",
    *,
    task_text: str = "schema_version: 1\n",
    result_text: str = '{"status":"completed"}\n',
) -> Path:
    staging = root / f"{name}.staging"
    (staging / "inputs").mkdir(parents=True)
    (staging / "handoff").mkdir()
    (staging / "inputs" / "task.yaml").write_text(
        task_text,
        encoding="utf-8",
    )
    (staging / "handoff" / "result.json").write_text(
        result_text,
        encoding="utf-8",
    )
    return staging


def test_publish_seals_staging_directory_and_verify_reads_it(
    tmp_path: Path,
) -> None:
    staging = _stage_sample_package(tmp_path)

    package_ref = publish(staging)
    release_dir = tmp_path / "f1-release-01"

    assert package_ref.package_dir == release_dir
    assert not staging.exists()
    manifest_bytes = (release_dir / "artifact_manifest.json").read_bytes()
    assert package_ref.manifest_sha256 == hashlib.sha256(manifest_bytes).hexdigest()

    verified = verify(package_ref)
    assert verified.read_text("handoff/result.json") == '{"status":"completed"}\n'
    assert verified.manifest_paths == {
        "handoff/result.json",
        "inputs/task.yaml",
    }


def test_verify_rejects_file_tampering_after_publication(tmp_path: Path) -> None:
    package_ref = publish(_stage_sample_package(tmp_path))
    (package_ref.package_dir / "handoff" / "result.json").write_text(
        '{"status":"COMPLETED"}\n',
        encoding="utf-8",
    )

    with pytest.raises(ArtifactError) as raised:
        verify(package_ref)

    assert raised.value.reason_code == "FILE_DIGEST_MISMATCH"
    assert raised.value.path == "handoff/result.json"


def test_verify_rejects_an_unmanifested_extra_file(tmp_path: Path) -> None:
    package_ref = publish(_stage_sample_package(tmp_path))
    (package_ref.package_dir / "inputs" / "extra.txt").write_text(
        "not sealed\n",
        encoding="utf-8",
    )

    with pytest.raises(ArtifactError) as raised:
        verify(package_ref)

    assert raised.value.reason_code == "FILE_SET_MISMATCH"


def test_verify_rejects_a_whole_package_replacement(tmp_path: Path) -> None:
    original = publish(_stage_sample_package(tmp_path, "f1-release-01"))
    (original.package_dir / "inputs" / "task.yaml").unlink()
    forged_staging = _stage_sample_package(
        tmp_path,
        "f1-release-02",
        task_text="schema_version: 2\n",
    )
    forged = publish(forged_staging)

    forged_ref = PackageRef(
        package_dir=forged.package_dir,
        manifest_sha256=original.manifest_sha256,
    )
    with pytest.raises(ArtifactError) as raised:
        verify(forged_ref)

    assert raised.value.reason_code == "MANIFEST_DIGEST_MISMATCH"


def test_publish_refuses_to_overwrite_an_existing_release(tmp_path: Path) -> None:
    first = publish(_stage_sample_package(tmp_path, "f1-release-01"))

    with pytest.raises(ArtifactError) as raised:
        publish(_stage_sample_package(tmp_path, "f1-release-01"))

    assert raised.value.reason_code == "RELEASE_ALREADY_EXISTS"
    assert verify(first).manifest_paths == {
        "handoff/result.json",
        "inputs/task.yaml",
    }


def test_publish_rejects_symlinks_inside_the_package(tmp_path: Path) -> None:
    staging = _stage_sample_package(tmp_path)
    external_file = tmp_path / "external.txt"
    external_file.write_text("must not be packaged\n", encoding="utf-8")
    (staging / "inputs" / "linked.txt").symlink_to(external_file)

    with pytest.raises(ArtifactError) as raised:
        publish(staging)

    assert raised.value.reason_code == "SYMLINK_NOT_ALLOWED"
