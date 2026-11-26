import json
from pathlib import Path

from .f1 import F1TaskPack
from .intake_v2 import IntakeTaskPackV2, ModelAnalysisReport, SkillHandoffRequest, SkillHandoffReceipt
from .f2 import F2BehaviorContract
from .handoff import HandoffResult
from .stage import StageSpec
from .verdict import GateDecision, GateResult


def export_json_schemas(output_dir: str | Path) -> list[Path]:
    from core.artifacts import ArtifactManifest

    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)

    models = {
        "artifact-manifest.schema.json": ArtifactManifest,
        "f1-task-pack.schema.json": F1TaskPack,
        "intake-task-pack-v2.schema.json": IntakeTaskPackV2,
        "model-analysis-v2.schema.json": ModelAnalysisReport,
        "skill-handoff-request-v2.schema.json": SkillHandoffRequest,
        "skill-handoff-receipt-v2.schema.json": SkillHandoffReceipt,
        "f2-behavior-contract.schema.json": F2BehaviorContract,
        "gate-decision.schema.json": GateDecision,
        "gate-result.schema.json": GateResult,
        "handoff-result.schema.json": HandoffResult,
        "stage-spec.schema.json": StageSpec,
    }

    exported: list[Path] = []
    for filename, model in models.items():
        path = directory / filename
        payload = json.dumps(
            model.model_json_schema(),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        path.write_text(payload + "\n", encoding="utf-8")
        exported.append(path)
    return exported
