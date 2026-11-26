"""Builders for `step_anatomy.yaml`, `behavior_contract.yaml`, and `observation_plan.yaml`."""

from __future__ import annotations

from core.contracts.anatomy_spec import ActionSpec, AnatomySpec, PI05_SPEC, ViewSpec
from typing import Any

from core.contracts.anatomy import (
    AnatomyBehaviorContract,
    AnatomyCaseSpec,
    AssertionSpec,
    BatchSpec,
    OptionalViewsSpec,
    PhaseSpec,
    PrecisionSpec,
    SegmentViewSpec,
    ValidatorSpec,
)

OBSERVATION_PLAN_BOUNDARIES: tuple[dict[str, str], ...] = (
    {"name": "init_weights", "mode": "exact"},
    {"name": "distributed_ranks", "mode": "exact"},
    {"name": "batch_sample_order", "mode": "exact"},
    {"name": "camera_role_content_map", "mode": "exact"},
    {"name": "rng_consumer_events", "mode": "exact"},
    {"name": "forward_call_count", "mode": "exact"},
    {"name": "optimizer_real_weight_update", "mode": "exact"},
    {"name": "numerical_endpoints", "mode": "exact_or_non_deterministic_flag"},
    {"name": "action_evidence", "mode": "exact"},
)
OBSERVATION_PLAN_NEGATIVE_INJECTIONS: tuple[str, ...] = (
    "skip_forward",
    "swap_cameras",
    "swap_samples",
    "swap_rng_consumers",
    "step_counter_only",
    "missing_rank_evidence",
    "swap_action_tokens",
    "swap_action_masks",
)

__all__ = [
    "ActionSpec", "AnatomySpec", "ViewSpec", "PI05_SPEC", "observation_boundaries",
    "negative_injections", "build_step_anatomy", "build_behavior_contract", "build_observation_plan",
]


def observation_boundaries(spec: AnatomySpec) -> list[dict[str, str]]:
    return [
        dict(item) for item in OBSERVATION_PLAN_BOUNDARIES
        if not (item["name"] == "camera_role_content_map" and not any(view.enabled for view in spec.views))
        and not (item["name"] == "rng_consumer_events" and not spec.required_rng_consumers)
    ]


def negative_injections(spec: AnatomySpec) -> list[str]:
    return [
        name for name in OBSERVATION_PLAN_NEGATIVE_INJECTIONS
        if not (name == "swap_cameras" and not any(view.enabled for view in spec.views))
        and not (name == "swap_rng_consumers" and not spec.required_rng_consumers)
        and not (name == "swap_action_tokens" and spec.action.representation != "discrete_tokens")
    ]


def build_step_anatomy(
    *,
    case_id: str,
    intake_manifest_sha256: str,
    upstream_commit: str,
    framework_commit: str,
    attempt1: dict[str, Any],
    attempt2: dict[str, Any],
    spec: AnatomySpec = PI05_SPEC,
) -> dict[str, Any]:
    """Build factual `analysis/step_anatomy.yaml` from verified attempt captures.

    Intentionally excludes unverifiable historical self-reported OS PIDs; independent
    subprocess isolation is proven by `AnatomyChecker` fresh dual-subprocess replay.
    """
    _ = attempt2
    return {
        "schema_version": 1,
        "anatomy_spec_sha256": spec.sha256(),
        "spec_manifest_sha256": attempt1.get("spec_manifest_sha256"),
        "case_id": case_id,
        "intake_manifest_sha256": intake_manifest_sha256,
        "upstream_commit": upstream_commit,
        "framework_commit": framework_commit,
        "execution_environment": attempt1["execution_environment"],
        "evidence_scope": attempt1["evidence_scope"],
        "adapter_id": attempt1["adapter_id"],
        "world_size": attempt1["world_size"],
        "seed": attempt1["seed"],
        "precision": dict(attempt1["precision"]),
        "sample_ids": list(attempt1["sample_ids"]),
        "cameras": attempt1["cameras"],
        "rng_events": attempt1["rng_events"],
        "action_evidence": attempt1["action_evidence"],
        "phases": attempt1["phases"],
        "forward_call_count": attempt1["forward_call_count"],
        "step_transition": {
            "before": attempt1["step_before"],
            "after": attempt1["step_after"],
        },
        "loss": attempt1["loss"],
        "grad_global_norm": attempt1["grad_global_norm"],
        "weights_before_digest": attempt1["weights_before_digest"],
        "weights_after_digest": attempt1["weights_after_digest"],
        "changed_parameters": list(attempt1["changed_parameters"]),
    }


def build_behavior_contract(
    *,
    case_id: str,
    intake_manifest_sha256: str,
    upstream_commit: str,
    framework_commit: str,
    attempt1: dict[str, Any],
    validator_id: str,
    validator_version: str,
    validator_sha256: str,
    unknowns: list[str] | None = None,
    blocked_reasons: list[str] | None = None,
    spec: AnatomySpec = PI05_SPEC,
) -> AnatomyBehaviorContract:
    """Build strict Pydantic `AnatomyBehaviorContract` for `analysis/behavior_contract.yaml`."""
    precision_doc = attempt1.get("precision", {})
    backend = precision_doc.get(
        "backend",
        "fake" if attempt1["execution_environment"] == "cpu_mock" else "jax",
    )
    gpu_count = 0 if backend == "fake" else int(attempt1.get("world_size", 1))

    return AnatomyBehaviorContract(
        schema_version=1,
        case=AnatomyCaseSpec(
            id=case_id,
            f1_manifest_sha256=intake_manifest_sha256,
            upstream_commit=upstream_commit,
            framework_commit=framework_commit,
            gpu_count=gpu_count,
            seed=int(attempt1["seed"]),
            batch=BatchSpec(sample_ids=list(attempt1["sample_ids"])),
            precision=PrecisionSpec(
                dtype=precision_doc.get("dtype", "float32"),
                math_mode=precision_doc.get("math_mode", "float32"),
                backend=backend,
                jit=bool(precision_doc.get("jit", False)),
            ),
            anatomy_spec_sha256=spec.sha256(),
            spec_manifest_sha256=attempt1.get("spec_manifest_sha256"),
        ),
        phases=[
            PhaseSpec(id="dataload", required=True),
            PhaseSpec(id="model_forward", required=True),
            PhaseSpec(id="loss", required=True),
            PhaseSpec(id="backward", required=True),
            PhaseSpec(id="optimizer_step", required=True),
        ],
        assertions=[
            AssertionSpec(id="sample-and-camera-order", type="state_digest", required=True),
            AssertionSpec(id="rng-consumer-order", type="rng_fingerprint", required=True),
            AssertionSpec(id="model-forward-count", type="call_count", required=True),
            AssertionSpec(id="loss-and-grad-endpoints", type="tensor_summary", required=True),
            AssertionSpec(id="real-parameter-update", type="state_digest", required=True),
            AssertionSpec(id="action-semantics", type="state_digest", required=True),
        ],
        optional_views=OptionalViewsSpec(
            segment_view=SegmentViewSpec(enabled=False, path=None)
        ),
        validator=ValidatorSpec(
            id=validator_id,
            version=validator_version,
            sha256=validator_sha256,
        ),
        unknowns=list(unknowns or []),
        blocked_reasons=list(blocked_reasons or []),
    )


def build_observation_plan(
    *,
    case_id: str,
    intake_manifest_sha256: str,
    attempt1: dict[str, Any],
    validator_id: str,
    validator_sha256: str,
    spec: AnatomySpec = PI05_SPEC,
) -> dict[str, Any]:
    """Build `analysis/observation_plan.yaml` declaring boundaries, comparators, and negative matrix."""
    return {
        "schema_version": 1,
        "anatomy_spec_sha256": spec.sha256(),
        "spec_manifest_sha256": attempt1.get("spec_manifest_sha256"),
        "case_id": case_id,
        "intake_manifest_sha256": intake_manifest_sha256,
        "execution_environment": attempt1["execution_environment"],
        "evidence_scope": attempt1["evidence_scope"],
        "adapter_id": attempt1["adapter_id"],
        "precision": dict(attempt1["precision"]),
        "comparator": "core.stages.f2_anatomy.compare:compare_attempts",
        "weight_diff_policy": "name -> shape -> dtype -> value (diff_weight_probe hierarchy)",
        "validator_id": validator_id,
        "validator_sha256": validator_sha256,
        "required_camera_roles": list(spec.required_camera_roles),
        "optional_camera_roles": list(spec.optional_camera_roles),
        "required_rng_consumers": list(spec.required_rng_consumers),
        "action": spec.to_dict()["action"],
        "views": spec.to_dict()["views"],
        "boundaries": observation_boundaries(spec),
        "negative_injections": negative_injections(spec),
    }
