"""Stage 1 (`intake`): reference model source lock, discovery, asset triage, probe, and freeze."""

from __future__ import annotations

from pathlib import Path

from core.stages.base import StageContext, StageResult

from .assets import AssetRecord, AssetTriageReport, triage_assets
from .discovery import ConfigCandidate, DiscoveryError, DiscoveryResult, EntrypointCandidate, discover_entrypoints
from .freeze import FreezeResult, analyze_intake, freeze_intake, init_preparation
from .probe import ProbeResult, run_startup_probe, validate_probe_argv
from .source import (
    SourceBundleInfo,
    SourceError,
    SourceSnapshot,
    SubmoduleRecord,
    WorktreeStatus,
    create_recursive_source_bundles,
    create_source_bundle,
    materialize_readonly_source_copy,
    restore_from_bundle_manifest,
    snapshot_source,
    verify_source_bundle,
    verify_source_unchanged,
)


class IntakeStageRunner:
    """StageRunner implementation executing the full offline Stage 1 (`intake`) loop."""

    stage_name = "intake"

    def run(self, context: StageContext) -> StageResult:
        task_path_raw = context.parameters.get("task_path")
        if not task_path_raw:
            raise ValueError("IntakeStageRunner requires parameters['task_path']")

        probe_mode = context.parameters.get("probe_mode", "none")
        prep_dir = Path(context.output_dir)
        init_preparation(
            Path(task_path_raw),
            prep_dir,
            preparation_id=context.run_id,
            request_id=str(context.parameters.get("request_id", f"{context.run_id}-req")),
        )
        analyze_intake(
            prep_dir,
            probe_mode=probe_mode,
            repo_root=context.workspace_dir,
            timeout_sec=float(context.parameters.get("timeout_sec", 60.0)),
            handoff_dir=context.parameters.get("handoff_dir"),
        )
        freeze_res = freeze_intake(
            prep_dir,
            release_name=context.parameters.get("release_name"),
            attempt_id=str(context.parameters.get("attempt_id", "intake-attempt-01")),
        )
        return StageResult(
            stage="intake",
            status=freeze_res.status,
            summary=freeze_res.summary,
            artifact_dir=freeze_res.package_ref.package_dir
            if freeze_res.package_ref is not None
            else None,
            manifest_sha256=freeze_res.package_ref.manifest_sha256
            if freeze_res.package_ref is not None
            else None,
            reason_codes=freeze_res.reason_codes,
            metadata={
                "release_record_path": str(freeze_res.release_record_path)
                if freeze_res.release_record_path is not None
                else None,
                "observed_scope": list(freeze_res.observed_scope),
            },
        )


__all__ = [
    "AssetRecord",
    "AssetTriageReport",
    "ConfigCandidate",
    "DiscoveryError",
    "DiscoveryResult",
    "EntrypointCandidate",
    "FreezeResult",
    "IntakeStageRunner",
    "ProbeResult",
    "SourceBundleInfo",
    "SourceError",
    "SourceSnapshot",
    "SubmoduleRecord",
    "WorktreeStatus",
    "analyze_intake",
    "create_recursive_source_bundles",
    "create_source_bundle",
    "discover_entrypoints",
    "freeze_intake",
    "init_preparation",
    "materialize_readonly_source_copy",
    "restore_from_bundle_manifest",
    "run_startup_probe",
    "snapshot_source",
    "triage_assets",
    "validate_probe_argv",
    "verify_source_bundle",
    "verify_source_unchanged",
]
