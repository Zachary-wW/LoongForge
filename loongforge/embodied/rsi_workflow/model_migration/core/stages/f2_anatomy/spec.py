"""Self-contained pre-capture releases, checked against a caller-held trust anchor."""
from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from typing import Any

import yaml

from core.artifacts import PackageRef, publish, verify
from core.contracts.common import load_yaml_document
from .contract import AnatomySpec

SPEC_RELEASE_REL = "evidence/spec_freeze"


def load_anatomy_spec(path: str | Path) -> AnatomySpec:
    """Load a strict bare policy or a schema-2 Intake/frozen-spec wrapper.

    Wrapper metadata is not a trust anchor; the caller still verifies the
    sealed release, source binding and external manifest digest separately.
    """
    document = load_yaml_document(path)
    if "spec" not in document:
        return AnatomySpec.from_dict(document)
    layouts = (
        {"schema_version", "spec"},
        {"schema_version", "spec", "source_ref", "source_commit", "request_id"},
        {"schema_version", "spec", "source_ref", "anatomy_spec_sha256",
         "intake_manifest_sha256", "upstream_commit"},
    )
    if (type(document.get("schema_version")) is not int
            or document["schema_version"] != 2 or set(document) not in layouts):
        raise ValueError("AnatomySpec wrapper must use a declared schema-2 layout")
    spec = AnatomySpec.from_dict(document["spec"])
    if "anatomy_spec_sha256" in document and document["anatomy_spec_sha256"] != spec.sha256():
        raise ValueError("AnatomySpec wrapper policy digest mismatch")
    return spec


def freeze_spec(
    prep: Path, spec: AnatomySpec, *, intake_manifest_sha256: str, upstream_commit: str,
    source_ref: dict[str, Any] | None = None,
) -> tuple[Path, dict[str, Any], str]:
    """Seal before execution and return the digest to the trusted Controller."""
    document = {
        "schema_version": 2, "spec": spec.to_dict(), "anatomy_spec_sha256": spec.sha256(),
        "intake_manifest_sha256": intake_manifest_sha256, "upstream_commit": upstream_commit,
        "source_ref": source_ref,
    }
    lock_path = prep / "spec_lock.json"
    source = prep / "specs/frozen"
    if lock_path.exists():
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
        if lock.get("package_dir") != "specs/frozen":
            raise ValueError("preparation spec lock must use its fixed relative path")
        ref = PackageRef(package_dir=source, manifest_sha256=lock["manifest_sha256"])
        sealed = verify(ref)
        if load_yaml_document(ref.package_dir / "anatomy_spec.yaml") != document:
            raise ValueError("cannot change the pre-capture frozen spec")
    else:
        staging = prep / "specs" / "frozen.staging"
        staging.mkdir(parents=True)
        (staging / "anatomy_spec.yaml").write_text(
            yaml.safe_dump(document, sort_keys=False), encoding="utf-8",
        )
        ref = publish(staging)
        sealed = verify(ref)
        lock = {"package_dir": "specs/frozen", "manifest_sha256": ref.manifest_sha256}
        lock_path.write_text(json.dumps(lock, sort_keys=True) + "\n", encoding="utf-8")
    analysis = prep / "work" / "analysis"
    analysis.mkdir(parents=True, exist_ok=True)
    embedded = prep / "work" / SPEC_RELEASE_REL
    if embedded.exists():
        verify(PackageRef(package_dir=embedded, manifest_sha256=ref.manifest_sha256)).read_text("anatomy_spec.yaml")
    else:
        shutil.copytree(source, embedded)
    (analysis / "anatomy_spec.yaml").write_text(sealed.read_text("anatomy_spec.yaml"), encoding="utf-8")
    (analysis / "spec_lock.json").write_text(json.dumps({
        "package_dir": SPEC_RELEASE_REL, "manifest_sha256": ref.manifest_sha256,
    }, sort_keys=True) + "\n", encoding="utf-8")
    return embedded / "anatomy_spec.yaml", document, ref.manifest_sha256


def verify_frozen_spec(
    package_dir: Path, *, intake_manifest_sha256: str, upstream_commit: str,
    trusted_spec_manifest_sha256: str | None,
) -> tuple[AnatomySpec, Path]:
    """An embedded, self-declared lock is NOT a trust root."""
    if (not isinstance(trusted_spec_manifest_sha256, str)
            or re.fullmatch(r"[0-9a-f]{64}", trusted_spec_manifest_sha256) is None
            or trusted_spec_manifest_sha256 == "0" * 64):
        raise ValueError("external trusted spec_manifest_sha256 is required")
    doc = load_yaml_document(package_dir / "analysis/anatomy_spec.yaml")
    spec = load_anatomy_spec(package_dir / "analysis/anatomy_spec.yaml")
    lock = json.loads((package_dir / "analysis/spec_lock.json").read_text(encoding="utf-8"))
    if lock.get("package_dir") != SPEC_RELEASE_REL or lock.get("manifest_sha256") != trusted_spec_manifest_sha256:
        raise ValueError("embedded spec lock disagrees with the external trusted spec manifest")
    embedded = package_dir / SPEC_RELEASE_REL
    if not embedded.resolve().is_relative_to(package_dir.resolve()):
        raise ValueError("embedded spec release escapes package")
    ref = PackageRef(package_dir=embedded, manifest_sha256=trusted_spec_manifest_sha256)
    sealed = verify(ref)
    if load_yaml_document(ref.package_dir / "anatomy_spec.yaml") != doc:
        raise ValueError("packaged spec differs from pre-capture anchor")
    if (doc.get("anatomy_spec_sha256") != spec.sha256()
            or doc.get("intake_manifest_sha256") != intake_manifest_sha256
            or doc.get("upstream_commit") != upstream_commit):
        raise ValueError("spec digest/Stage 1 identity binding mismatch")
    # Force verification to consume the manifested file, not an unmanifested path.
    sealed.read_text("anatomy_spec.yaml")
    return spec, ref.package_dir / "anatomy_spec.yaml"
