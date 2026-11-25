from __future__ import annotations

import hashlib
import json
import os
import stat
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Literal

from pydantic import Field, model_validator

from core.contracts.common import Sha256, StrictModel

MANIFEST_NAME = "artifact_manifest.json"
STAGING_SUFFIX = ".staging"


class ArtifactError(ValueError):
    def __init__(
        self,
        reason_code: str,
        message: str,
        *,
        path: str | None = None,
    ) -> None:
        self.reason_code = reason_code
        self.message = message
        self.path = path
        super().__init__(message)

    def to_dict(self) -> dict[str, Any]:
        return {
            "reason_code": self.reason_code,
            "message": self.message,
            "path": self.path,
        }


class ManifestFile(StrictModel):
    path: str = Field(min_length=1)
    size: int = Field(ge=0)
    sha256: Sha256

    @model_validator(mode="after")
    def path_must_be_package_relative(self) -> "ManifestFile":
        if "\\" in self.path:
            raise ValueError("manifest paths must use POSIX separators")
        relative_path = PurePosixPath(self.path)
        if (
            relative_path.is_absolute()
            or not relative_path.parts
            or any(part in {"", ".", ".."} for part in relative_path.parts)
        ):
            raise ValueError("manifest paths must be normalized and relative")
        if self.path == MANIFEST_NAME:
            raise ValueError("manifest cannot index itself")
        return self


class ArtifactManifest(StrictModel):
    schema_version: Literal[1]
    files: list[ManifestFile] = Field(min_length=1)

    @model_validator(mode="after")
    def files_must_be_sorted_and_unique(self) -> "ArtifactManifest":
        paths = [item.path for item in self.files]
        if paths != sorted(paths):
            raise ValueError("manifest files must be sorted by path")
        if len(paths) != len(set(paths)):
            raise ValueError("manifest file paths must be unique")
        return self

    def encoded(self) -> bytes:
        payload = self.model_dump(mode="json")
        return (
            json.dumps(
                payload,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        ).encode("utf-8")


@dataclass(frozen=True, slots=True)
class PackageRef:
    package_dir: Path
    manifest_sha256: str


@dataclass(frozen=True, slots=True)
class VerifiedPackage:
    ref: PackageRef
    manifest: ArtifactManifest

    @property
    def manifest_paths(self) -> set[str]:
        return {item.path for item in self.manifest.files}

    def read_bytes(self, relative_path: str) -> bytes:
        self._require_manifested_path(relative_path)
        return (self.ref.package_dir / relative_path).read_bytes()

    def read_text(self, relative_path: str) -> str:
        return self.read_bytes(relative_path).decode("utf-8")

    def _require_manifested_path(self, relative_path: str) -> None:
        if relative_path not in self.manifest_paths:
            raise ArtifactError(
                "PATH_NOT_MANIFESTED",
                "path is not present in the verified manifest",
                path=relative_path,
            )


def publish(staging_dir: str | Path) -> PackageRef:
    staging = Path(staging_dir)
    _require_directory(staging, "STAGING_NOT_DIRECTORY")
    if not staging.name.endswith(STAGING_SUFFIX) or staging.name == STAGING_SUFFIX:
        raise ArtifactError(
            "INVALID_STAGING_NAME",
            f"staging directory must end with {STAGING_SUFFIX}",
            path=staging.name,
        )

    manifest_path = staging / MANIFEST_NAME
    if manifest_path.exists() or manifest_path.is_symlink():
        raise ArtifactError(
            "MANIFEST_ALREADY_PRESENT",
            "staging directory already contains an artifact manifest",
            path=MANIFEST_NAME,
        )

    files = _manifest_files(staging)
    manifest = ArtifactManifest(schema_version=1, files=files)
    manifest_bytes = manifest.encoded()

    destination = staging.with_name(staging.name[: -len(STAGING_SUFFIX)])
    if destination.exists() or destination.is_symlink():
        raise ArtifactError(
            "RELEASE_ALREADY_EXISTS",
            "release directory already exists",
            path=destination.name,
        )

    temporary_manifest = staging / f".{MANIFEST_NAME}.tmp"
    try:
        with temporary_manifest.open("xb") as stream:
            stream.write(manifest_bytes)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_manifest, manifest_path)
        _fsync_directory(staging)
        os.rename(staging, destination)
        _fsync_directory(destination.parent)
    finally:
        if temporary_manifest.exists():
            temporary_manifest.unlink()

    return PackageRef(
        package_dir=destination,
        manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
    )


def verify(ref: PackageRef) -> VerifiedPackage:
    package_dir = Path(ref.package_dir)
    _require_directory(package_dir, "PACKAGE_NOT_DIRECTORY")

    manifest_path = package_dir / MANIFEST_NAME
    _require_regular_file(manifest_path, "MANIFEST_NOT_REGULAR_FILE")
    manifest_bytes = manifest_path.read_bytes()
    actual_manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
    if actual_manifest_sha256 != ref.manifest_sha256:
        raise ArtifactError(
            "MANIFEST_DIGEST_MISMATCH",
            "artifact manifest does not match the trusted reference",
            path=MANIFEST_NAME,
        )

    try:
        manifest = ArtifactManifest.model_validate_json(manifest_bytes)
    except Exception as error:
        raise ArtifactError(
            "MANIFEST_INVALID",
            "artifact manifest is invalid",
            path=MANIFEST_NAME,
        ) from error

    actual_files = _manifest_files(package_dir)
    expected_paths = [item.path for item in manifest.files]
    actual_paths = [item.path for item in actual_files]
    if actual_paths != expected_paths:
        raise ArtifactError(
            "FILE_SET_MISMATCH",
            "package file set differs from the artifact manifest",
            path=MANIFEST_NAME,
        )

    expected_by_path = {item.path: item for item in manifest.files}
    for item in actual_files:
        expected = expected_by_path[item.path]
        if item.size != expected.size:
            raise ArtifactError(
                "FILE_SIZE_MISMATCH",
                "file size differs from the artifact manifest",
                path=item.path,
            )
        if item.sha256 != expected.sha256:
            raise ArtifactError(
                "FILE_DIGEST_MISMATCH",
                "file digest differs from the artifact manifest",
                path=item.path,
            )

    return VerifiedPackage(ref=ref, manifest=manifest)


def _manifest_files(package_dir: Path) -> list[ManifestFile]:
    files: list[ManifestFile] = []
    for root, directory_names, file_names in os.walk(
        package_dir,
        topdown=True,
        followlinks=False,
    ):
        root_path = Path(root)
        for directory_name in list(directory_names):
            child = root_path / directory_name
            if child.is_symlink():
                raise ArtifactError(
                    "SYMLINK_NOT_ALLOWED",
                    "symlinked directories are not allowed in a package",
                    path=child.relative_to(package_dir).as_posix(),
                )
            if not child.is_dir():
                raise ArtifactError(
                    "NON_REGULAR_DIRECTORY",
                    "package entries must be regular directories",
                    path=child.relative_to(package_dir).as_posix(),
                )

        for file_name in file_names:
            child = root_path / file_name
            relative_path = child.relative_to(package_dir).as_posix()
            if relative_path == MANIFEST_NAME:
                continue
            if child.is_symlink():
                raise ArtifactError(
                    "SYMLINK_NOT_ALLOWED",
                    "symlinked files are not allowed in a package",
                    path=relative_path,
                )
            _require_regular_file(child, "NON_REGULAR_FILE", relative_path)
            files.append(
                ManifestFile(
                    path=relative_path,
                    size=child.stat().st_size,
                    sha256=_hash_file(child),
                )
            )
    files.sort(key=lambda item: item.path)
    if not files:
        raise ArtifactError(
            "EMPTY_PACKAGE",
            "package must contain at least one manifested file",
        )
    return files


def _require_directory(path: Path, reason_code: str) -> None:
    try:
        path_stat = path.lstat()
    except FileNotFoundError as error:
        raise ArtifactError(
            reason_code, "path does not exist", path=str(path)
        ) from error
    if not stat.S_ISDIR(path_stat.st_mode):
        raise ArtifactError(reason_code, "path is not a directory", path=str(path))


def _require_regular_file(
    path: Path,
    reason_code: str,
    display_path: str | None = None,
) -> None:
    try:
        path_stat = path.lstat()
    except FileNotFoundError as error:
        raise ArtifactError(
            reason_code, "path does not exist", path=display_path
        ) from error
    if not stat.S_ISREG(path_stat.st_mode):
        raise ArtifactError(
            reason_code,
            "path is not a regular file",
            path=display_path or str(path),
        )


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fsync_directory(path: Path) -> None:
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
