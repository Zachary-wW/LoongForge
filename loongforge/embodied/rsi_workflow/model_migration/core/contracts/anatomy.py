"""Unified Stage 2 (`anatomy`) contract exports.

Re-exports the strict anatomy behavior contract models while keeping legacy
`f2.py` 100% backwards-compatible.
"""

from .f2 import (
    AssertionSpec,
    AssertionType,
    Backend,
    BatchSpec,
    F2BehaviorContract,
    F2CaseSpec,
    MathMode,
    OptionalViewsSpec,
    PhaseSpec,
    PrecisionDType,
    PrecisionSpec,
    SegmentViewSpec,
    ValidatorSpec,
    load_f2_behavior_contract,
)

AnatomyCaseSpec = F2CaseSpec
AnatomyBehaviorContract = F2BehaviorContract
load_anatomy_behavior_contract = load_f2_behavior_contract

__all__ = [
    "AnatomyBehaviorContract",
    "AnatomyCaseSpec",
    "AssertionSpec",
    "AssertionType",
    "Backend",
    "BatchSpec",
    "F2BehaviorContract",
    "F2CaseSpec",
    "MathMode",
    "OptionalViewsSpec",
    "PhaseSpec",
    "PrecisionDType",
    "PrecisionSpec",
    "SegmentViewSpec",
    "ValidatorSpec",
    "load_anatomy_behavior_contract",
    "load_f2_behavior_contract",
]
