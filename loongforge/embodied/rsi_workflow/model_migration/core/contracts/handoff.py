from pathlib import PurePosixPath
from typing import Any, Literal, TypeAlias

from pydantic import Field, model_validator

from .common import Sha256, StrictModel, load_yaml_document
from .errors import ContractValidationError

Stage: TypeAlias = Literal["f1", "f2"]
ExecutionStatus: TypeAlias = Literal["completed", "waiting", "blocked", "error"]
ObservedScope: TypeAlias = Literal[
    "static",
    "startup_probe",
    "train_step_replay",
    "checkpoint_resume",
    "task_eval",
]


class EvidenceRef(StrictModel):
    path: str = Field(min_length=1)
    sha256: Sha256

    @model_validator(mode="after")
    def path_must_be_relative(self) -> "EvidenceRef":
        relative_path = PurePosixPath(self.path)
        if (
            "\\" in self.path
            or relative_path.is_absolute()
            or any(part in {"", ".", ".."} for part in relative_path.parts)
        ):
            raise ValueError("evidence paths must be normalized and relative")
        return self


class HandoffResult(StrictModel):
    schema_version: Literal[1]
    preparation_id: str = Field(min_length=1)
    request_id: str = Field(min_length=1)
    stage: Stage
    attempt_id: str = Field(min_length=1)
    request_sha256: Sha256
    execution_status: ExecutionStatus
    observed_scope: list[ObservedScope] = Field(min_length=1)
    validator_verdict: EvidenceRef | None
    unknowns: list[str]
    blocked_reasons: list[str]
    anatomy_spec_sha256: Sha256 | None = None
    spec_manifest_sha256: Sha256 | None = None

    @model_validator(mode="after")
    def completed_requires_validator_verdict(self) -> "HandoffResult":
        if self.execution_status == "completed" and self.validator_verdict is None:
            raise ValueError("completed handoff requires a validator verdict")
        if self.execution_status in {"blocked", "error"} and not self.blocked_reasons:
            raise ValueError("blocked handoff requires at least one reason")
        return self


def load_handoff_result(path: Any) -> HandoffResult:
    try:
        return HandoffResult.model_validate(load_yaml_document(path))
    except ContractValidationError:
        raise
    except Exception as error:
        validation_error = getattr(error, "errors", None)
        if callable(validation_error):
            raise ContractValidationError(validation_error()) from error
        raise
