from typing import Any, Literal
from pathlib import PurePosixPath

from pydantic import Field, field_validator, model_validator

from .common import (
    AssetMode,
    AssetSpec,
    CommitHash,
    StrictModel,
    load_yaml_document,
)
from .errors import ContractValidationError


class SourceSpec(StrictModel):
    repo: str = Field(min_length=1)
    commit: CommitHash
    model_variant: str = Field(min_length=1)
    behavior_reference: str | None = None


class TargetSpec(StrictModel):
    framework: str = Field(min_length=1)
    commit: CommitHash


class EnvironmentSpec(StrictModel):
    name: str = Field(min_length=1)
    gpu_count: int = Field(ge=0)
    image: str | None = None
    dependencies_lock: str | None = None
    python: str | None = None


class EntrypointSpec(StrictModel):
    path: str = Field(min_length=1)
    argv: list[str] = Field(min_length=1)
    cwd: str
    env: dict[str, str]
    topology: str = Field(min_length=1)

    @field_validator("path")
    @classmethod
    def path_must_be_relative_and_normalized(cls, value: str) -> str:
        relative_path = PurePosixPath(value)
        if (
            "\\" in value
            or relative_path.is_absolute()
            or relative_path.as_posix() != value
            or any(part in {"", ".", ".."} for part in relative_path.parts)
        ):
            raise ValueError("entrypoint path must be relative and normalized")
        return value


class TaskScope(StrictModel):
    static: bool
    startup_probe: bool
    train_step: bool
    checkpoint_resume: bool
    task_eval: bool
    asset_mode: AssetMode


class F1TaskPack(StrictModel):
    schema_version: Literal[1]
    source: SourceSpec
    target: TargetSpec
    environment: EnvironmentSpec
    entrypoint: EntrypointSpec
    assets: list[AssetSpec]
    scope: TaskScope
    unknowns: list[str]
    blocked_reasons: list[str]

    @model_validator(mode="after")
    def asset_ids_must_be_unique(self) -> "F1TaskPack":
        asset_ids = [asset.id for asset in self.assets]
        if len(asset_ids) != len(set(asset_ids)):
            raise ValueError("asset ids must be unique")
        return self


def load_f1_task_pack(path: Any) -> F1TaskPack:
    try:
        document = load_yaml_document(path)
        if document.get("schema_version") == 2:
            from .intake_v2 import IntakeTaskPackV2
            return IntakeTaskPackV2.model_validate(document)
        return F1TaskPack.model_validate(document)
    except ContractValidationError:
        raise
    except Exception as error:
        validation_error = getattr(error, "errors", None)
        if callable(validation_error):
            raise ContractValidationError(validation_error()) from error
        raise
