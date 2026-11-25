"""Layered attempt comparator and hierarchical weight-probe diff for Stage 2 (`anatomy`).

Hierarchical weight diff follows the exact 4-tier priority from
`loongforge-skills/.../precision-alignment/tools/diff_weight_probe.py`:
  1. name-set differences
  2. shape mismatches
  3. dtype mismatches
  4. value (sha256) mismatches (grouped by first 4 dot-separated name components)
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any, Literal, TypeAlias

from .contract import REQUIRED_CAMERA_ROLES, REQUIRED_RNG_CONSUMERS

CompareStatus: TypeAlias = Literal["pass", "fail", "non_deterministic"]
REQUIRED_PHASE_IDS: tuple[str, ...] = (
    "dataload",
    "model_forward",
    "loss",
    "backward",
    "optimizer_step",
)


@dataclass(frozen=True, slots=True)
class ComparisonReport:
    """Deterministic comparison result across two single-step attempts."""

    status: CompareStatus
    summary: str
    reason_codes: tuple[str, ...] = ()
    first_mismatch: str | None = None
    expected: Any = None
    actual: Any = None
    evidence_path: str | None = None
    weight_diff_before: dict[str, Any] | None = None
    weight_diff_after: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "status": self.status,
            "summary": self.summary,
            "reason_codes": list(self.reason_codes),
            "first_mismatch": self.first_mismatch,
            "expected": self.expected,
            "actual": self.actual,
            "evidence_path": self.evidence_path,
            "weight_diff_before": self.weight_diff_before,
            "weight_diff_after": self.weight_diff_after,
        }


def diff_weight_snapshots(
    base: Mapping[str, Mapping[str, Any]],
    candidate: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Hierarchical name -> shape -> dtype -> value diff (modeled on `diff_weight_probe.py`)."""
    base_keys = set(base.keys())
    cand_keys = set(candidate.keys())
    only_a = sorted(base_keys - cand_keys)
    only_b = sorted(cand_keys - base_keys)
    common = sorted(base_keys & cand_keys)

    shape_diff: list[dict[str, Any]] = []
    dtype_diff: list[dict[str, Any]] = []
    value_diff: list[dict[str, Any]] = []

    for name in common:
        entry_a = base[name]
        entry_b = candidate[name]
        if entry_a.get("shape") != entry_b.get("shape"):
            shape_diff.append(
                {"name": name, "base": entry_a.get("shape"), "candidate": entry_b.get("shape")}
            )
        elif entry_a.get("dtype") != entry_b.get("dtype"):
            dtype_diff.append(
                {"name": name, "base": entry_a.get("dtype"), "candidate": entry_b.get("dtype")}
            )
        elif entry_a.get("sha256") != entry_b.get("sha256"):
            value_diff.append(
                {"name": name, "base": entry_a.get("sha256"), "candidate": entry_b.get("sha256")}
            )

    value_groups: dict[str, int] = defaultdict(int)
    for item in value_diff:
        prefix = ".".join(str(item["name"]).split(".")[:4])
        value_groups[prefix] += 1

    first_category: str | None = None
    if only_a or only_b:
        first_category = "name"
    elif shape_diff:
        first_category = "shape"
    elif dtype_diff:
        first_category = "dtype"
    elif value_diff:
        first_category = "value"

    return {
        "identical": first_category is None,
        "first_category": first_category,
        "base_count": len(base),
        "candidate_count": len(candidate),
        "common_count": len(common),
        "only_a": only_a,
        "only_b": only_b,
        "shape_mismatches": shape_diff,
        "dtype_mismatches": dtype_diff,
        "value_mismatches": value_diff,
        "value_mismatch_groups": dict(sorted(value_groups.items())),
    }


def compare_attempts(
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
    *,
    evidence_path: str = "evidence/attempt-2/capture.json",
    required_camera_roles: Iterable[str] = REQUIRED_CAMERA_ROLES,
    required_rng_consumers: Iterable[str] = REQUIRED_RNG_CONSUMERS,
) -> ComparisonReport:
    """Compare two single-step attempt captures in deterministic causal priority order.

    Priority order:
      0. Execution metadata & initial state parity (adapter, case, execution_environment,
         evidence_scope, step_before, rng_initial_state)
      1. Initial parameter snapshot identity (name -> shape -> dtype -> value)
      2. Distributed rank coverage (`world_size` vs `rank_receipts`, non-empty)
      3. Input sample ordering (`sample_ids`, non-empty)
      4. Camera slot/content/mask mapping (`cameras`, non-empty)
      5. Per-consumer RNG events (`rng_events`, non-empty)
      6. Forward call count & required `phases` execution (non-empty)
      7. Real parameter update & step progression (`step_before -> step_after`, `changed_parameters`)
      8. Post-step structural weight parity (name/shape/dtype -> `fail`)
      9. Numerical endpoints (`loss`, `grad_global_norm`, weight values) -> `non_deterministic`
    """
    # 0. Execution metadata & initial state parity across both attempts.
    for meta_key in (
        "adapter_id",
        "case_id",
        "execution_environment",
        "evidence_scope",
        "precision",
        "world_size",
        "seed",
        "step_before",
        "rng_initial_state",
    ):
        if baseline.get(meta_key) != candidate.get(meta_key):
            return ComparisonReport(
                status="fail",
                summary=f"attempt metadata/initial state mismatch on {meta_key}",
                reason_codes=("ATTEMPT_METADATA_MISMATCH",),
                first_mismatch=f"init.{meta_key}",
                expected=baseline.get(meta_key),
                actual=candidate.get(meta_key),
                evidence_path=evidence_path,
            )

    # 1. Initial weights must be identical before the step runs.
    w_before_base = baseline.get("weights_before", {})
    w_before_cand = candidate.get("weights_before", {})
    if not w_before_base or not w_before_cand:
        return ComparisonReport(
            status="fail",
            summary="initial weights snapshot is empty in one or both attempts",
            reason_codes=("MISSING_INITIAL_WEIGHTS",),
            first_mismatch="init.weights_before",
            expected="non-empty weights_before mapping",
            actual={
                "attempt1_count": len(w_before_base),
                "attempt2_count": len(w_before_cand),
            },
            evidence_path=evidence_path,
        )
    w_before_diff = diff_weight_snapshots(w_before_base, w_before_cand)
    if not w_before_diff["identical"]:
        return ComparisonReport(
            status="fail",
            summary="initial parameter snapshot differs before optimizer step",
            reason_codes=("INITIAL_WEIGHTS_MISMATCH",),
            first_mismatch=f"init.weights_before:{w_before_diff['first_category']}",
            expected=baseline.get("weights_before_digest"),
            actual=candidate.get("weights_before_digest"),
            evidence_path=evidence_path,
            weight_diff_before=w_before_diff,
        )

    # 2. Per-rank coverage check (multi-rank cannot report only rank 0).
    expected_world = int(baseline.get("world_size", 1))
    if expected_world < 1:
        return ComparisonReport(
            status="fail",
            summary=f"invalid world_size: {expected_world}",
            reason_codes=("MISSING_RANK_EVIDENCE",),
            first_mismatch="distributed.ranks",
            expected="world_size >= 1",
            actual=expected_world,
            evidence_path=evidence_path,
            weight_diff_before=w_before_diff,
        )
    expected_ranks = list(range(expected_world))
    for label, record, ev_path in (
        ("attempt-1", baseline, "evidence/attempt-1/capture.json"),
        ("attempt-2", candidate, evidence_path),
    ):
        receipts = record.get("rank_receipts")
        if not isinstance(receipts, list) or not receipts:
            return ComparisonReport(
                status="fail",
                summary=f"{label} is missing per-rank execution receipts",
                reason_codes=("MISSING_RANK_EVIDENCE",),
                first_mismatch="distributed.ranks",
                expected=expected_ranks,
                actual=receipts,
                evidence_path=ev_path,
                weight_diff_before=w_before_diff,
            )
        actual_ranks = sorted(int(item["rank"]) for item in receipts)
        if actual_ranks != expected_ranks or len(receipts) != len(expected_ranks):
            return ComparisonReport(
                status="fail",
                summary=f"{label} is missing per-rank execution receipts",
                reason_codes=("MISSING_RANK_EVIDENCE",),
                first_mismatch="distributed.ranks",
                expected=expected_ranks,
                actual=actual_ranks,
                evidence_path=ev_path,
                weight_diff_before=w_before_diff,
            )

    # 3. Sample order exact match (must be non-empty in both attempts).
    base_samples = list(baseline.get("sample_ids", []))
    cand_samples = list(candidate.get("sample_ids", []))
    if not base_samples or not cand_samples:
        return ComparisonReport(
            status="fail",
            summary="sample_ids evidence is missing in one or both attempts",
            reason_codes=("MISSING_SAMPLE_EVIDENCE",),
            first_mismatch="input.sample_ids",
            expected="non-empty sample_ids",
            actual={"attempt1": base_samples, "attempt2": cand_samples},
            evidence_path=evidence_path,
            weight_diff_before=w_before_diff,
        )
    if base_samples != cand_samples:
        return ComparisonReport(
            status="fail",
            summary="batch sample_ids ordering mismatch between attempts",
            reason_codes=("SAMPLE_ORDER_MISMATCH",),
            first_mismatch="input.sample_ids",
            expected=base_samples,
            actual=cand_samples,
            evidence_path=evidence_path,
            weight_diff_before=w_before_diff,
        )

    # 4. Camera role -> content/shape/dtype/mask mapping exact match + frozen required roles.
    req_cam_roles = tuple(required_camera_roles)
    base_cams = baseline.get("cameras", {})
    cand_cams = candidate.get("cameras", {})
    if not isinstance(base_cams, Mapping) or not isinstance(cand_cams, Mapping) or not base_cams or not cand_cams:
        return ComparisonReport(
            status="fail",
            summary="camera role/content evidence is missing in one or both attempts",
            reason_codes=("MISSING_CAMERA_EVIDENCE",),
            first_mismatch="input.cameras",
            expected=list(req_cam_roles),
            actual={"attempt1": base_cams, "attempt2": cand_cams},
            evidence_path=evidence_path,
            weight_diff_before=w_before_diff,
        )
    for label, cams_map, ev_path in (
        ("attempt-1", base_cams, "evidence/attempt-1/capture.json"),
        ("attempt-2", cand_cams, evidence_path),
    ):
        missing_roles = [role for role in req_cam_roles if role not in cams_map]
        if missing_roles:
            return ComparisonReport(
                status="fail",
                summary=f"{label} is missing required camera role(s): {', '.join(missing_roles)}",
                reason_codes=("MISSING_CAMERA_ROLE",),
                first_mismatch=f"input.cameras.{missing_roles[0]}",
                expected=list(req_cam_roles),
                actual=sorted(cams_map.keys()),
                evidence_path=ev_path,
                weight_diff_before=w_before_diff,
            )
    if base_cams != cand_cams:
        first_cam = next(
            (
                key
                for key in sorted(set(base_cams) | set(cand_cams))
                if base_cams.get(key) != cand_cams.get(key)
            ),
            "unknown",
        )
        return ComparisonReport(
            status="fail",
            summary=f"camera role/content mapping mismatch at {first_cam}",
            reason_codes=("CAMERA_ORDER_MISMATCH",),
            first_mismatch=f"input.cameras.{first_cam}",
            expected=base_cams.get(first_cam),
            actual=cand_cams.get(first_cam),
            evidence_path=evidence_path,
            weight_diff_before=w_before_diff,
        )

    # 5. Per-consumer RNG events exact match + frozen required consumers.
    req_rng_consumers = tuple(required_rng_consumers)
    base_rng = list(baseline.get("rng_events", []))
    cand_rng = list(candidate.get("rng_events", []))
    if not base_rng or not cand_rng:
        return ComparisonReport(
            status="fail",
            summary="per-consumer RNG events are missing in one or both attempts",
            reason_codes=("MISSING_RNG_EVIDENCE",),
            first_mismatch="rng.events",
            expected=list(req_rng_consumers),
            actual={"attempt1": base_rng, "attempt2": cand_rng},
            evidence_path=evidence_path,
            weight_diff_before=w_before_diff,
        )
    for label, rng_list, ev_path in (
        ("attempt-1", base_rng, "evidence/attempt-1/capture.json"),
        ("attempt-2", cand_rng, evidence_path),
    ):
        present_consumers = {
            item.get("consumer")
            for item in rng_list
            if isinstance(item, Mapping) and isinstance(item.get("consumer"), str)
        }
        missing_consumers = [c for c in req_rng_consumers if c not in present_consumers]
        if missing_consumers:
            return ComparisonReport(
                status="fail",
                summary=f"{label} is missing required RNG consumer(s): {', '.join(missing_consumers)}",
                reason_codes=("MISSING_RNG_CONSUMER",),
                first_mismatch=f"rng.events.{missing_consumers[0]}",
                expected=list(req_rng_consumers),
                actual=sorted(present_consumers),
                evidence_path=ev_path,
                weight_diff_before=w_before_diff,
            )
    if base_rng != cand_rng:
        first_idx = next(
            (
                idx
                for idx in range(max(len(base_rng), len(cand_rng)))
                if (base_rng[idx] if idx < len(base_rng) else None)
                != (cand_rng[idx] if idx < len(cand_rng) else None)
            ),
            0,
        )
        return ComparisonReport(
            status="fail",
            summary=f"RNG consumer/draw event mismatch at index {first_idx}",
            reason_codes=("RNG_CONSUMER_MISMATCH",),
            first_mismatch=f"rng.events[{first_idx}]",
            expected=base_rng[first_idx] if first_idx < len(base_rng) else None,
            actual=cand_rng[first_idx] if first_idx < len(cand_rng) else None,
            evidence_path=evidence_path,
            weight_diff_before=w_before_diff,
        )

    # 6. Forward call count (cannot skip forward or fake with cached outputs).
    base_fwd = int(baseline.get("forward_call_count", 0))
    cand_fwd = int(candidate.get("forward_call_count", 0))
    if base_fwd < 1 or cand_fwd < 1 or base_fwd != cand_fwd:
        return ComparisonReport(
            status="fail",
            summary=f"forward call count mismatch or skipped (base={base_fwd}, cand={cand_fwd})",
            reason_codes=("FORWARD_CALL_SKIPPED",),
            first_mismatch="forward.call_count",
            expected=max(1, base_fwd),
            actual=cand_fwd,
            evidence_path=evidence_path,
            weight_diff_before=w_before_diff,
        )

    # 7. Real optimizer step & parameter update check (rejects step_counter_only).
    for label, record, ev_path in (
        ("attempt-1", baseline, "evidence/attempt-1/capture.json"),
        ("attempt-2", candidate, evidence_path),
    ):
        step_delta = int(record.get("step_after", 0)) - int(record.get("step_before", 0))
        if step_delta != 1:
            return ComparisonReport(
                status="fail",
                summary=f"{label} did not advance step counter by exactly 1 (delta={step_delta})",
                reason_codes=("STEP_DELTA_INVALID",),
                first_mismatch="optimizer.step_delta",
                expected=1,
                actual=step_delta,
                evidence_path=ev_path,
                weight_diff_before=w_before_diff,
            )
        w_before_rec = record.get("weights_before", {})
        w_after_rec = record.get("weights_after", {})
        if not isinstance(w_after_rec, Mapping) or not w_after_rec:
            return ComparisonReport(
                status="fail",
                summary=f"{label} has empty weights_after snapshot; cannot prove parameter update",
                reason_codes=("MISSING_WEIGHTS_AFTER",),
                first_mismatch="optimizer.weights_after",
                expected=sorted(w_before_rec.keys()),
                actual={},
                evidence_path=ev_path,
                weight_diff_before=w_before_diff,
            )
        if set(w_before_rec.keys()) != set(w_after_rec.keys()):
            return ComparisonReport(
                status="fail",
                summary=f"{label} weights_after parameter keys differ from weights_before",
                reason_codes=("WEIGHTS_AFTER_KEYS_MISMATCH",),
                first_mismatch="optimizer.weights_after",
                expected=sorted(w_before_rec.keys()),
                actual=sorted(w_after_rec.keys()),
                evidence_path=ev_path,
                weight_diff_before=w_before_diff,
            )
        w_self_diff = diff_weight_snapshots(w_before_rec, w_after_rec)
        actual_changed = sorted(
            name
            for name in w_before_rec
            if w_before_rec[name].get("sha256") != w_after_rec[name].get("sha256")
        )
        changed = list(record.get("changed_parameters", []))
        if (
            w_self_diff["identical"]
            or not actual_changed
            or sorted(changed) != actual_changed
            or not record.get("optimizer_applied")
        ):
            return ComparisonReport(
                status="fail",
                summary=f"{label} lacks verified real parameter weight updates",
                reason_codes=("NO_REAL_PARAMETER_UPDATE",),
                first_mismatch="optimizer.weight_update",
                expected={"actual_changed_parameters": actual_changed or "non-empty"},
                actual={
                    "optimizer_applied": record.get("optimizer_applied"),
                    "changed_parameters": changed,
                    "actual_changed_parameters": actual_changed,
                    "weights_identical": w_self_diff["identical"],
                },
                evidence_path=ev_path,
                weight_diff_before=w_before_diff,
            )

    # 7b. Required phases completeness & cross-attempt phase trace parity.
    base_phases = baseline.get("phases")
    cand_phases = candidate.get("phases")
    for label, phase_list, ev_path in (
        ("attempt-1", base_phases, "evidence/attempt-1/capture.json"),
        ("attempt-2", cand_phases, evidence_path),
    ):
        if not isinstance(phase_list, list) or not phase_list:
            return ComparisonReport(
                status="fail",
                summary=f"{label} is missing required execution phases evidence",
                reason_codes=("MISSING_PHASE_EVIDENCE",),
                first_mismatch="phases",
                expected=list(REQUIRED_PHASE_IDS),
                actual=phase_list,
                evidence_path=ev_path,
                weight_diff_before=w_before_diff,
            )
        phase_map = {
            item.get("id"): item
            for item in phase_list
            if isinstance(item, Mapping) and "id" in item
        }
        missing_or_unexecuted = [
            pid
            for pid in REQUIRED_PHASE_IDS
            if pid not in phase_map
            or not phase_map[pid].get("executed")
            or int(phase_map[pid].get("call_count", 0)) < 1
        ]
        if missing_or_unexecuted:
            return ComparisonReport(
                status="fail",
                summary=f"{label} missing or unexecuted required phase(s): {', '.join(missing_or_unexecuted)}",
                reason_codes=("PHASE_NOT_EXECUTED",),
                first_mismatch=f"phases.{missing_or_unexecuted[0]}",
                expected=list(REQUIRED_PHASE_IDS),
                actual=phase_list,
                evidence_path=ev_path,
                weight_diff_before=w_before_diff,
            )
    if base_phases != cand_phases:
        return ComparisonReport(
            status="fail",
            summary="phase execution trace differs between attempt-1 and attempt-2",
            reason_codes=("PHASE_TRACE_MISMATCH",),
            first_mismatch="phases",
            expected=base_phases,
            actual=cand_phases,
            evidence_path=evidence_path,
            weight_diff_before=w_before_diff,
        )

    # 8. Post-step structural weight parity (name / shape / dtype mismatch -> deterministic `fail`).
    w_after_diff = diff_weight_snapshots(
        baseline.get("weights_after", {}),
        candidate.get("weights_after", {}),
    )
    if not w_after_diff["identical"] and w_after_diff["first_category"] in {
        "name",
        "shape",
        "dtype",
    }:
        cat = str(w_after_diff["first_category"])
        return ComparisonReport(
            status="fail",
            summary=f"post-step weight structural mismatch ({cat})",
            reason_codes=("WEIGHT_STRUCTURE_MISMATCH",),
            first_mismatch=f"weights_after.{cat}",
            expected=baseline.get("weights_after"),
            actual=candidate.get("weights_after"),
            evidence_path=evidence_path,
            weight_diff_before=w_before_diff,
            weight_diff_after=w_after_diff,
        )

    # 9. Numerical endpoints: loss, grad norm, and post-update weight values.
    # Only numerical value divergence (with identical causal inputs & tensor structure)
    # is classified as `non_deterministic` — never disguised as `pass`.
    if baseline.get("loss") != candidate.get("loss"):
        return ComparisonReport(
            status="non_deterministic",
            summary="loss endpoint drifted across identical single-step attempts",
            reason_codes=("NON_DETERMINISTIC_ENDPOINT",),
            first_mismatch="numerical.loss",
            expected=baseline.get("loss"),
            actual=candidate.get("loss"),
            evidence_path=evidence_path,
            weight_diff_before=w_before_diff,
            weight_diff_after=w_after_diff,
        )
    if baseline.get("grad_global_norm") != candidate.get("grad_global_norm"):
        return ComparisonReport(
            status="non_deterministic",
            summary="gradient global norm drifted across identical single-step attempts",
            reason_codes=("NON_DETERMINISTIC_ENDPOINT",),
            first_mismatch="numerical.grad_global_norm",
            expected=baseline.get("grad_global_norm"),
            actual=candidate.get("grad_global_norm"),
            evidence_path=evidence_path,
            weight_diff_before=w_before_diff,
            weight_diff_after=w_after_diff,
        )
    if not w_after_diff["identical"]:
        return ComparisonReport(
            status="non_deterministic",
            summary="post-step weight values drifted across identical attempts",
            reason_codes=("NON_DETERMINISTIC_ENDPOINT",),
            first_mismatch="numerical.weights_after:value",
            expected=baseline.get("weights_after_digest"),
            actual=candidate.get("weights_after_digest"),
            evidence_path=evidence_path,
            weight_diff_before=w_before_diff,
            weight_diff_after=w_after_diff,
        )

    return ComparisonReport(
        status="pass",
        summary="two independent single-step attempts matched across all exact and numerical boundaries",
        reason_codes=(),
        first_mismatch=None,
        expected=None,
        actual=None,
        evidence_path=None,
        weight_diff_before=w_before_diff,
        weight_diff_after=w_after_diff,
    )
