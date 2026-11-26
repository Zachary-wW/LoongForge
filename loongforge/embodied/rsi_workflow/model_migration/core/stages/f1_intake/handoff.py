"""Read-only, deterministic v2 handoff import and re-verification helpers."""
from __future__ import annotations

import ast
import hashlib
import json
import shutil
import tempfile
from pathlib import Path
from typing import Any

import yaml

from core.contracts.intake_v2 import (
    CodeLocation, IntakeTaskPackV2, ModelAnalysisReport, PaperLocation,
    SkillHandoffReceipt,
)
from .source import _run_git, verify_source_bundle

V2_ANALYSIS_FILES = (
    "model_analysis.json", "slot_hits.json", "training_parameters.yaml",
    "slot_catalog.yaml", "skill_handoff_receipt.json", "anatomy_spec.yaml",
)
V2_PACKAGE_FILES = tuple(f"analysis/{x}" for x in V2_ANALYSIS_FILES) + ("evidence/skill.bundle",)


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")).hexdigest()


def _inside(root: Path, relative: str) -> Path:
    path = root / relative
    if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"handoff path escapes permitted directory: {relative}")
    return path


def _symbols(tree: ast.AST) -> dict[str, list[ast.AST]]:
    symbols: dict[str, list[ast.AST]] = {}

    def visit(node: ast.AST, prefix: str = "") -> None:
        child_prefix = prefix
        names = []
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            names = [node.name]
            child_prefix = f"{prefix}{node.name}."
        elif isinstance(node, ast.Assign):
            names = [ast.unparse(x) for x in node.targets]
        elif isinstance(node, ast.AnnAssign):
            names = [ast.unparse(node.target)]
        for name in names:
            symbols.setdefault(prefix + name, []).append(node)
        for child in ast.iter_child_nodes(node):
            visit(child, child_prefix)
    visit(tree)
    return symbols


def verify_code_location(location: CodeLocation, task: IntakeTaskPackV2, repo: Path) -> ast.AST:
    if location.commit != task.source.commit:
        raise ValueError("code evidence commit differs from pinned source")
    if not any(location.path == p or location.path.startswith(p + "/") for p in task.analysis_handoff.read_paths):
        raise ValueError("code evidence is outside handoff read permissions")
    data = _run_git(repo, "show", f"{location.commit}:{location.path}", binary=True).stdout
    if hashlib.sha256(data).hexdigest() != location.blob_sha256:
        raise ValueError("code evidence blob SHA mismatch")
    tree = ast.parse(data, filename=location.path)
    nodes = _symbols(tree).get(location.symbol, [])
    hits = [n for n in nodes if n.lineno == location.line_start and n.end_lineno == location.line_end]
    if len(hits) != 1:
        raise ValueError("code evidence symbol/line range does not match pinned AST")
    return hits[0]


def _parameter_value(source: CodeLocation | PaperLocation, task: IntakeTaskPackV2, repo: Path, evidence: Path) -> Any:
    if isinstance(source, CodeLocation):
        node = verify_code_location(source, task, repo)
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            raise ValueError("parameter source must identify a literal assignment")
        return ast.literal_eval(node.value)
    data = _inside(evidence, source.path).read_bytes()
    if hashlib.sha256(data).hexdigest() != source.sha256:
        raise ValueError("paper evidence SHA mismatch")
    text = data.decode("utf-8")
    if any(value not in text for value in (source.paper_id, source.version, source.section, source.excerpt)):
        raise ValueError("paper version/section/excerpt is not present in supplied text evidence")
    # A textual evidence adapter only: never infer a value from an unlocated paper citation.
    if source.value_literal not in source.excerpt or f"page: {source.page}" not in text:
        raise ValueError("paper default value/page is not located in its excerpt")
    return json.loads(source.value_literal)


def validate_handoff(
    task: IntakeTaskPackV2, report_path: Path, *, repo: Path, evidence_root: Path,
) -> tuple[ModelAnalysisReport, SkillHandoffReceipt]:
    request = task.analysis_handoff
    raw = report_path.read_bytes()
    report_sha = hashlib.sha256(raw).hexdigest()
    if report_sha != request.report_sha256:
        raise ValueError("handoff report SHA mismatch")
    report = ModelAnalysisReport.model_validate(json.loads(raw))
    if (report.status != "completed" or report.request_id != request.request_id
            or report.source_commit != task.source.commit or report.target_commit != task.target.commit):
        raise ValueError("handoff report is partial/failed or has an identity mismatch")
    catalog_sha = canonical_sha256(task.slot_catalog.model_dump(mode="json"))
    if catalog_sha != request.catalog_sha256:
        raise ValueError("slot catalog SHA mismatch")
    catalog = {x.extension_point: x for x in task.slot_catalog.extensions}
    for slot in report.slots:
        target = catalog[slot.extension_point]
        if slot.side != target.side or not set(slot.target_files).issubset(target.target_files):
            raise ValueError("slot mapping differs from declared target layout")
        for source in slot.sources:
            verify_code_location(source, task, repo)
    for parameter in report.training_parameters:
        defaults = [_parameter_value(source, task, repo, evidence_root) for source in parameter.sources]
        if any(canonical_sha256(x) != canonical_sha256(defaults[parameter.selected_source]) for x in defaults) and not (
            parameter.resolution and parameter.resolution.strip()
        ):
            raise ValueError("conflicting code/paper defaults require an explicit resolution")
        if canonical_sha256(defaults[parameter.selected_source]) != canonical_sha256(parameter.default_value):
            raise ValueError("parameter default disagrees with independently read source")
        effective = (
            _parameter_value(parameter.effective_source, task, repo, evidence_root)
            if parameter.effective_source is not None else parameter.default_value
        )
        if canonical_sha256(effective) != canonical_sha256(parameter.effective_value):
            raise ValueError("parameter effective value disagrees with declared override source")
    verify_code_location(report.anatomy_spec_source.source_ref, task, repo)
    # Bind the actual playbook bytes to an offline pinned Skill Git tree, not a self-reported SHA.
    bundle = _inside(evidence_root, "evidence/skill.bundle")
    with tempfile.TemporaryDirectory(prefix="intake-skill-verify-") as tmp:
        restored = Path(tmp) / "skill"
        verify_source_bundle(
            bundle, expected_commit=request.skill.commit,
            expected_sha256=request.skill.bundle_sha256, restore_dir=restored,
        )
        playbook = _run_git(
            restored, "show", f"{request.skill.commit}:{request.skill.playbook_path}", binary=True,
        ).stdout
        if hashlib.sha256(playbook).hexdigest() != request.skill.playbook_sha256:
            raise ValueError("Skill playbook SHA differs from pinned Skill commit")
    return report, SkillHandoffReceipt(
        schema_version=2, request_id=request.request_id, boundary="handoff_only",
        status="imported", harness_executed=False,
        request_sha256=canonical_sha256(request.model_dump(mode="json")),
        report_sha256=report_sha, catalog_sha256=catalog_sha,
        skill_commit=request.skill.commit, playbook_sha256=request.skill.playbook_sha256,
        anatomy_spec_sha256=report.anatomy_spec.sha256(),
        anatomy_spec_source_sha256=canonical_sha256(report.anatomy_spec_source.model_dump(mode="json")),
    )


def import_handoff(task: IntakeTaskPackV2, handoff_dir: Path, work: Path, *, repo: Path) -> None:
    report_path = _inside(handoff_dir, "model_analysis.json")
    report, receipt = validate_handoff(task, report_path, repo=repo, evidence_root=handoff_dir)
    work.mkdir(parents=True, exist_ok=True)
    (work / "model_analysis.json").write_bytes(report_path.read_bytes())
    documents = {
        "slot_hits.json": [x.model_dump(mode="json") for x in report.slots],
        "training_parameters.yaml": [x.model_dump(mode="json") for x in report.training_parameters],
        "slot_catalog.yaml": task.slot_catalog.model_dump(mode="json"),
        "skill_handoff_receipt.json": receipt.model_dump(mode="json"),
        "anatomy_spec.yaml": spec_document(report),
    }
    for name, document in documents.items():
        text = json.dumps(document, ensure_ascii=False, indent=2) if name.endswith(".json") else yaml.safe_dump(document)
        (work / name).write_text(text + "\n", encoding="utf-8")
    paths = {"evidence/skill.bundle"}
    for param in report.training_parameters:
        for source in [*param.sources, param.effective_source]:
            if isinstance(source, PaperLocation):
                paths.add(source.path)
    for path in paths:
        destination = work / path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(_inside(handoff_dir, path), destination)


def recheck_handoff(task: IntakeTaskPackV2, package: Path, repo: Path) -> dict[str, str]:
    """Independent Checker path: rebuild receipt and derived files from original report."""
    report, receipt = validate_handoff(
        task, package / "analysis/model_analysis.json", repo=repo, evidence_root=package,
    )
    expected = {
        "slot_hits.json": [x.model_dump(mode="json") for x in report.slots],
        "training_parameters.yaml": [x.model_dump(mode="json") for x in report.training_parameters],
        "slot_catalog.yaml": task.slot_catalog.model_dump(mode="json"),
        "skill_handoff_receipt.json": receipt.model_dump(mode="json"),
        "anatomy_spec.yaml": spec_document(report),
    }
    for filename, document in expected.items():
        path = package / "analysis" / filename
        actual = json.loads(path.read_text()) if filename.endswith(".json") else yaml.safe_load(path.read_text())
        if canonical_sha256(actual) != canonical_sha256(document):
            raise ValueError(f"v2 derived artifact differs from independent reconstruction: {filename}")
    return {name: hashlib.sha256((package / name).read_bytes()).hexdigest() for name in V2_PACKAGE_FILES}


def spec_document(report: ModelAnalysisReport) -> dict[str, Any]:
    return {
        "schema_version": 2, "spec": report.anatomy_spec.to_dict(),
        "source_ref": report.anatomy_spec_source.model_dump(mode="json"),
        "source_commit": report.source_commit, "request_id": report.request_id,
    }
