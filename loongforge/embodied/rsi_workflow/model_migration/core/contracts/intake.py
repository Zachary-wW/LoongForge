"""Unified Stage 1 (`intake`) contract exports.

Re-exports the strict intake task pack models while keeping legacy `f1.py`
100% backwards-compatible.
"""

from .f1 import (
    EntrypointSpec,
    EnvironmentSpec,
    F1TaskPack,
    SourceSpec,
    TargetSpec,
    TaskScope,
    load_f1_task_pack,
)

IntakeTaskPack = F1TaskPack
load_intake_task_pack = load_f1_task_pack

__all__ = [
    "EntrypointSpec",
    "EnvironmentSpec",
    "F1TaskPack",
    "IntakeTaskPack",
    "SourceSpec",
    "TargetSpec",
    "TaskScope",
    "load_f1_task_pack",
    "load_intake_task_pack",
]
