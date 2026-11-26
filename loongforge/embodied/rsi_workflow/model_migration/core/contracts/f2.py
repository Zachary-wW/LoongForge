from typing import Any, Literal, TypeAlias

from pydantic import Field, model_validator

from .common import CommitHash, Sha256, StrictModel, load_yaml_document
from .errors import ContractValidationError

PrecisionDType: TypeAlias = Literal["bfloat16", "float16", "float32"]
MathMode: TypeAlias = Literal["float32", "tf32"]
Backend: TypeAlias = Literal["fake", "jax", "pytorch"]
AssertionType: TypeAlias = Literal[
    "call_count",
    "rng_fingerprint",
    "tensor_summary",
    "state_digest",
]


class BatchSpec(StrictModel):
    sample_ids: list[str] = Field(min_length=1)


class PrecisionSpec(StrictModel):
    dtype: PrecisionDType
    math_mode: MathMode
    backend: Backend
    jit: bool


class F2CaseSpec(StrictModel):
    id: str = Field(min_length=1)
    f1_manifest_sha256: Sha256
    upstream_commit: CommitHash
    framework_commit: CommitHash
    gpu_count: int = Field(ge=0)
    seed: int = Field(ge=0)
    batch: BatchSpec
    precision: PrecisionSpec
    anatomy_spec_sha256: Sha256 | None = None
    spec_manifest_sha256: Sha256 | None = None


class PhaseSpec(StrictModel):
    id: str = Field(min_length=1)
    required: bool


class AssertionSpec(StrictModel):
    id: str = Field(min_length=1)
    type: AssertionType
    required: bool


class SegmentViewSpec(StrictModel):
    enabled: bool
    path: str | None


class OptionalViewsSpec(StrictModel):
    segment_view: SegmentViewSpec


class ValidatorSpec(StrictModel):
    id: str = Field(min_length=1)
    version: str = Field(min_length=1)
    sha256: Sha256


class F2BehaviorContract(StrictModel):
    schema_version: Literal[1]
    case: F2CaseSpec
    phases: list[PhaseSpec] = Field(min_length=1)
    assertions: list[AssertionSpec] = Field(min_length=1)
    optional_views: OptionalViewsSpec
    validator: ValidatorSpec
    unknowns: list[str]
    blocked_reasons: list[str]

    @model_validator(mode="after")
    def phase_ids_must_be_unique(self) -> "F2BehaviorContract":
        phase_ids = [phase.id for phase in self.phases]
        if len(phase_ids) != len(set(phase_ids)):
            raise ValueError("phase ids must be unique")
        assertion_ids = [assertion.id for assertion in self.assertions]
        if len(assertion_ids) != len(set(assertion_ids)):
            raise ValueError("assertion ids must be unique")
        return self


def load_f2_behavior_contract(path: Any) -> F2BehaviorContract:
    try:
        return F2BehaviorContract.model_validate(load_yaml_document(path))
    except ContractValidationError:
        raise
    except Exception as error:
        validation_error = getattr(error, "errors", None)
        if callable(validation_error):
            raise ContractValidationError(validation_error()) from error
        raise
