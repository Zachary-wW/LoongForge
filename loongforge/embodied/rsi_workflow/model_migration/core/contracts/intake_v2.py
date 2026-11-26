"""Internal v2 skill handoff, soft engineering slots and parameter provenance.

The local pipeline imports a completed playbook report; it never calls a Python
Skill API or claims it launched an Agent harness.
"""
from __future__ import annotations

from pathlib import PurePosixPath
from typing import Annotated, Any, Literal

from pydantic import Field, JsonValue, field_validator, model_validator

from .common import CommitHash, Sha256, StrictModel
from .f1 import F1TaskPack
from .anatomy_spec import AnatomySpec

ExtensionPoint = Literal[
    "model_registry", "data_and_transforms", "config_map",
    "trainer_builder", "eval_adapter", "regression_runner",
]
Side = Literal["data", "model", "train"]
EXTENSION_POINTS = (
    "model_registry", "data_and_transforms", "config_map",
    "trainer_builder", "eval_adapter", "regression_runner",
)


def normalized_path(value: str) -> str:
    path = PurePosixPath(value)
    if (not value or "\\" in value or path.is_absolute()
            or path.as_posix() != value or any(x in {".", "..", ""} for x in path.parts)):
        raise ValueError("path must be normalized and relative")
    return value


class ExtensionTarget(StrictModel):
    extension_point: ExtensionPoint
    side: Side
    registry_path: str
    target_files: list[str] = Field(min_length=1)
    reuse_components: list[str] = Field(default_factory=list)

    _registry_path = field_validator("registry_path")(normalized_path)

    @field_validator("target_files", "reuse_components")
    @classmethod
    def paths(cls, values: list[str]) -> list[str]:
        if len(set(values)) != len(values):
            raise ValueError("duplicate target paths")
        return [normalized_path(x) for x in values]


class SlotCatalog(StrictModel):
    schema_version: Literal[1]
    compatibility_profile: Literal["internal"]
    version: str = Field(min_length=1)
    target_commit: CommitHash
    extensions: list[ExtensionTarget] = Field(min_length=6, max_length=6)

    @model_validator(mode="after")
    def all_extensions(self) -> SlotCatalog:
        if sorted(x.extension_point for x in self.extensions) != sorted(EXTENSION_POINTS):
            raise ValueError("catalog must declare each of the six real extension points once")
        if {x.side for x in self.extensions} != {"data", "model", "train"}:
            raise ValueError("catalog must cover data/model/train engineering groups")
        return self


class SkillIdentity(StrictModel):
    name: Literal["model-adaptation"]
    repo: str = Field(min_length=1)
    commit: CommitHash
    playbook_path: str
    playbook_sha256: Sha256
    bundle_sha256: Sha256

    _path = field_validator("playbook_path")(normalized_path)


class SkillHandoffRequest(StrictModel):
    schema_version: Literal[2]
    request_id: str = Field(min_length=1)
    boundary: Literal["handoff_only"]
    source_commit: CommitHash
    target_commit: CommitHash
    skill: SkillIdentity
    report_sha256: Sha256
    catalog_sha256: Sha256
    read_paths: list[str] = Field(min_length=1)
    write_paths: list[str] = Field(min_length=1)

    @field_validator("read_paths", "write_paths")
    @classmethod
    def paths(cls, values: list[str]) -> list[str]:
        if len(values) != len(set(values)):
            raise ValueError("duplicate permission paths")
        return [normalized_path(x) for x in values]

    @model_validator(mode="after")
    def output_boundary(self) -> SkillHandoffRequest:
        if any(x != "analysis" and not x.startswith("analysis/") for x in self.write_paths):
            raise ValueError("handoff output permission is confined to analysis/")
        return self


class CodeLocation(StrictModel):
    kind: Literal["code"]
    commit: CommitHash
    path: str
    blob_sha256: Sha256
    symbol: str = Field(min_length=1)
    line_start: int = Field(ge=1)
    line_end: int = Field(ge=1)

    _path = field_validator("path")(normalized_path)

    @model_validator(mode="after")
    def line_range(self) -> CodeLocation:
        if self.line_end < self.line_start:
            raise ValueError("invalid source line range")
        return self


class PaperLocation(StrictModel):
    kind: Literal["paper"]
    paper_id: str = Field(min_length=1)
    version: str = Field(min_length=1)
    section: str = Field(min_length=1)
    page: int = Field(ge=1)
    path: str
    sha256: Sha256
    excerpt: str = Field(min_length=1)
    value_literal: str = Field(min_length=1)

    @field_validator("path")
    @classmethod
    def evidence_path(cls, value: str) -> str:
        normalized_path(value)
        if not value.startswith("evidence/papers/"):
            raise ValueError("paper evidence must live under evidence/papers/")
        return value


ParameterSource = Annotated[CodeLocation | PaperLocation, Field(discriminator="kind")]


class SlotHit(StrictModel):
    slot_id: str = Field(min_length=1)
    extension_point: ExtensionPoint
    side: Side
    status: Literal["applicable", "not_applicable"]
    sources: list[CodeLocation]
    target_files: list[str]
    reason: str | None = None

    @field_validator("target_files")
    @classmethod
    def paths(cls, values: list[str]) -> list[str]:
        if len(set(values)) != len(values):
            raise ValueError("duplicate slot target paths")
        return [normalized_path(x) for x in values]

    @model_validator(mode="after")
    def applicability(self) -> SlotHit:
        if self.status == "applicable" and (not self.sources or not self.target_files):
            raise ValueError("applicable slot needs source evidence and target files")
        if self.status == "not_applicable" and (not self.reason or not self.reason.strip() or self.sources or self.target_files):
            raise ValueError("not_applicable slot needs reason and must not claim hits")
        return self


class TrainingParameter(StrictModel):
    name: str = Field(min_length=1)
    default_value: JsonValue
    effective_value: JsonValue
    sources: list[ParameterSource] = Field(min_length=1)
    selected_source: int = Field(ge=0)
    resolution: str | None = None
    effective_source: ParameterSource | None = None

    @model_validator(mode="after")
    def selected(self) -> TrainingParameter:
        if self.selected_source >= len(self.sources):
            raise ValueError("selected parameter source is out of range")
        if self.default_value != self.effective_value and self.effective_source is None:
            raise ValueError("overridden effective value needs its own source")
        return self


class AnatomySpecSource(StrictModel):
    source_ref: CodeLocation
    analysis_basis: str = Field(min_length=1)
    approved_by: str | None = None  # metadata, not an authenticated approval claim


class ModelAnalysisReport(StrictModel):
    schema_version: Literal[2]
    request_id: str = Field(min_length=1)
    status: Literal["completed", "partial", "failed"]
    source_commit: CommitHash
    target_commit: CommitHash
    slots: list[SlotHit] = Field(min_length=6)
    training_parameters: list[TrainingParameter] = Field(min_length=1)
    anatomy_spec: AnatomySpec
    anatomy_spec_source: AnatomySpecSource

    @field_validator("anatomy_spec", mode="before")
    @classmethod
    def typed_spec(cls, value: Any) -> AnatomySpec:
        return value if isinstance(value, AnatomySpec) else AnatomySpec.from_dict(value)

    @model_validator(mode="after")
    def unique_ids(self) -> ModelAnalysisReport:
        for ids in ([x.slot_id for x in self.slots], [x.name for x in self.training_parameters]):
            if len(ids) != len(set(ids)):
                raise ValueError("report IDs must be unique")
        if {x.extension_point for x in self.slots} != set(EXTENSION_POINTS):
            raise ValueError("report must account for all extension points")
        return self


class SkillHandoffReceipt(StrictModel):
    schema_version: Literal[2]
    request_id: str
    boundary: Literal["handoff_only"]
    status: Literal["imported"]
    harness_executed: Literal[False]
    request_sha256: Sha256
    report_sha256: Sha256
    catalog_sha256: Sha256
    skill_commit: CommitHash
    playbook_sha256: Sha256
    anatomy_spec_sha256: Sha256
    anatomy_spec_source_sha256: Sha256


class IntakeTaskPackV2(F1TaskPack):
    schema_version: Literal[2]
    analysis_handoff: SkillHandoffRequest
    slot_catalog: SlotCatalog

    @model_validator(mode="after")
    def handoff_identity(self) -> IntakeTaskPackV2:
        if (self.analysis_handoff.source_commit != self.source.commit
                or self.analysis_handoff.target_commit != self.target.commit
                or self.slot_catalog.target_commit != self.target.commit):
            raise ValueError("v2 task/handoff/catalog commit mismatch")
        return self
