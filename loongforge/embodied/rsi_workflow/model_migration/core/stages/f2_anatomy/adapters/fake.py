"""Deterministic pure-CPU `FakeStepAdapter` for Stage 2 (`anatomy`) offline verification.

Every output is explicitly tagged with:
- `execution_environment = "cpu_mock"`
- `evidence_scope = "orchestration_only"`
so mock CPU captures can never be mistaken for real GPU / real checkpoint evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, TypeAlias

from ..contract import AnatomySpec, PI05_SPEC
from ..spec import load_anatomy_spec

FakeNegativeInjection: TypeAlias = Literal[
    "skip_forward",
    "swap_cameras",
    "swap_samples",
    "swap_rng_consumers",
    "step_counter_only",
    "missing_rank_evidence",
    "swap_action_tokens",
    "swap_action_masks",
]

FAKE_NEGATIVE_INJECTIONS: tuple[FakeNegativeInjection, ...] = (
    "skip_forward",
    "swap_cameras",
    "swap_samples",
    "swap_rng_consumers",
    "step_counter_only",
    "missing_rank_evidence",
    "swap_action_tokens",
    "swap_action_masks",
)

DEFAULT_SAMPLE_IDS: tuple[str, ...] = (
    "libero_10:episode-000:frame-000",
    "libero_10:episode-001:frame-000",
)


@dataclass(frozen=True, slots=True)
class FakeStepConfig:
    """Deterministic configuration for one synthetic CPU optimizer step."""

    spec: AnatomySpec = PI05_SPEC
    seed: int = 42
    sample_ids: tuple[str, ...] = DEFAULT_SAMPLE_IDS
    world_size: int = 1
    learning_rate: float = 0.05
    dtype: str = "float32"
    inject: FakeNegativeInjection | None = None
    case_id: str = "pi05-cpu-mock-step-1"

    def __post_init__(self) -> None:
        if self.seed < 0:
            raise ValueError("seed must be non-negative")
        if self.world_size < 1:
            raise ValueError("world_size must be >= 1")
        if not self.sample_ids:
            raise ValueError("sample_ids must be non-empty")
        if self.inject is not None and self.inject not in FAKE_NEGATIVE_INJECTIONS:
            raise ValueError(
                f"unknown fake injection {self.inject!r}; expected one of {FAKE_NEGATIVE_INJECTIONS}"
            )


def _sha256_floats(values: list[float]) -> str:
    packed = struct.pack(f"<{len(values)}d", *values)
    return hashlib.sha256(packed).hexdigest()


def _sha256_json(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _tensor_summary(name: str, values: list[float], shape: list[int], dtype: str) -> dict[str, Any]:
    norm = math.sqrt(sum(v * v for v in values))
    mean = sum(values) / len(values)
    return {
        "name": name,
        "kind": "P",
        "shape": shape,
        "dtype": dtype,
        "sha256": _sha256_floats(values),
        "l2_norm": round(norm, 10),
        "mean": round(mean, 10),
        "finite": all(math.isfinite(v) for v in values),
    }


class FakeStepAdapter:
    """Deterministic single-step adapter with explicit 6-class negative injection hooks."""

    adapter_id = "fake_cpu_step_adapter"
    execution_environment = "cpu_mock"
    evidence_scope = "orchestration_only"

    def execute_step(self, config: FakeStepConfig) -> dict[str, Any]:
        """Execute one deterministic synthetic training step and emit full evidence."""
        # 1. Fixed initial parameters (identical across independent processes for same seed).
        init_rng = random.Random(config.seed * 1009 + 7)
        w_enc = [round(init_rng.uniform(-0.5, 0.5), 6) for _ in range(8)]
        w_head = [round(init_rng.uniform(-0.5, 0.5), 6) for _ in range(8)]
        weights_before = {
            "vision_encoder.weight": _tensor_summary(
                "vision_encoder.weight", w_enc, [2, 4], config.dtype
            ),
            "action_head.weight": _tensor_summary(
                "action_head.weight", w_head, [2, 4], config.dtype
            ),
        }
        weights_before_digest = _sha256_json(weights_before)

        # 2. Batch sample ordering + per-camera payloads.
        sample_ids = list(config.sample_ids)
        if config.inject == "swap_samples" and len(sample_ids) >= 2:
            sample_ids = [sample_ids[1], sample_ids[0], *sample_ids[2:]]

        cameras: dict[str, dict[str, Any]] = {}
        for slot, view in enumerate(config.spec.views):
            if not view.enabled:
                continue
            cameras[view.role] = {
                "slot": slot, "enabled": True, "mask": view.mask,
                "placeholder": view.placeholder,
                "shape": [len(sample_ids), *view.shape], "dtype": view.dtype,
                "sha256": view.content_digest(sample_ids),
            }
        if config.inject == "swap_cameras" and cameras:
            roles = list(cameras)
            if len(roles) > 1:
                x, y = roles[:2]
                cameras[x]["sha256"], cameras[y]["sha256"] = cameras[y]["sha256"], cameras[x]["sha256"]
            else:
                cameras[roles[0]]["sha256"] = hashlib.sha256(b"wrong-view").hexdigest()

        batch_digest = _sha256_json({"sample_ids": sample_ids, "cameras": cameras})

        # 3. Deterministic per-consumer RNG draws (preprocess / noise / time).
        step_rng = random.Random(config.seed * 7919 + 13)
        noise_draws = [round(step_rng.gauss(0.0, 1.0), 6) for _ in range(8)]
        time_draws = [round(step_rng.uniform(0.001, 1.0), 6) for _ in range(len(sample_ids))]

        rng_events = [
            {"order": order, "consumer": consumer,
             "key_id": f"step0:fold:{config.seed}:{order}",
             "draw_sha256": _sha256_floats(noise_draws if order % 2 == 0 else time_draws)}
            for order, consumer in enumerate(config.spec.required_rng_consumers)
        ]
        if config.inject == "swap_rng_consumers" and rng_events:
            if len(rng_events) > 1:
                rng_events[0]["consumer"], rng_events[1]["consumer"] = rng_events[1]["consumer"], rng_events[0]["consumer"]
            else:
                rng_events[0]["consumer"] = "undeclared_consumer"

        if not config.spec.required_rng_consumers:
            noise_draws = [0.0] * 8
            time_draws = [1.0] * len(sample_ids)

        # 4. Forward, loss, backward, optimizer update.
        # Derive deterministic target vector from sample_ids + cameras + noise + time.
        signal_seed = int(batch_digest[:8], 16) % 97 + 1
        targets = [
            round(noise_draws[i] * time_draws[i % len(time_draws)] + 0.01 * (i + signal_seed), 6)
            for i in range(8)
        ]

        if config.inject == "skip_forward":
            forward_call_count = 0
            preds = [0.0] * 8
        else:
            forward_call_count = 1
            preds = [round(w_enc[i] * 0.5 + w_head[i], 6) for i in range(8)]

        diffs = [preds[i] - targets[i] for i in range(8)]
        loss = round(sum(d * d for d in diffs) / len(diffs), 10)

        if config.inject == "skip_forward":
            grad_enc = [0.0] * 8
            grad_head = [0.0] * 8
        else:
            grad_enc = [round((2.0 / len(diffs)) * diffs[i] * 0.5, 8) for i in range(8)]
            grad_head = [round((2.0 / len(diffs)) * diffs[i], 8) for i in range(8)]

        gradients = {
            "vision_encoder.weight": _tensor_summary(
                "vision_encoder.weight.grad", grad_enc, [2, 4], config.dtype
            ),
            "action_head.weight": _tensor_summary(
                "action_head.weight.grad", grad_head, [2, 4], config.dtype
            ),
        }
        grad_global_norm = round(
            math.sqrt(sum(g * g for g in grad_enc) + sum(g * g for g in grad_head)),
            10,
        )

        if config.inject == "step_counter_only":
            w_enc_after = list(w_enc)
            w_head_after = list(w_head)
            optimizer_applied = False
        else:
            w_enc_after = [
                round(w_enc[i] - config.learning_rate * grad_enc[i], 8)
                for i in range(8)
            ]
            w_head_after = [
                round(w_head[i] - config.learning_rate * grad_head[i], 8)
                for i in range(8)
            ]
            optimizer_applied = True

        weights_after = {
            "vision_encoder.weight": _tensor_summary(
                "vision_encoder.weight", w_enc_after, [2, 4], config.dtype
            ),
            "action_head.weight": _tensor_summary(
                "action_head.weight", w_head_after, [2, 4], config.dtype
            ),
        }
        weights_after_digest = _sha256_json(weights_after)
        changed_params = sorted(
            name
            for name in weights_before
            if weights_before[name]["sha256"] != weights_after[name]["sha256"]
        )

        # 5. Distributed per-rank receipts.
        if config.inject == "missing_rank_evidence":
            reported_ranks = [0]
        else:
            reported_ranks = list(range(config.world_size))

        rank_receipts = [
            {
                "rank": r,
                "step_before": 0,
                "step_after": 1,
                "forward_call_count": forward_call_count,
                "weights_after_digest": weights_after_digest,
            }
            for r in reported_ranks
        ]

        phases = [
            {"id": "dataload", "executed": True, "call_count": 1},
            {
                "id": "model_forward",
                "executed": forward_call_count > 0,
                "call_count": forward_call_count,
            },
            {"id": "loss", "executed": forward_call_count > 0, "call_count": 1 if forward_call_count > 0 else 0},
            {"id": "backward", "executed": forward_call_count > 0, "call_count": 1 if forward_call_count > 0 else 0},
            {
                "id": "optimizer_step",
                "executed": optimizer_applied and len(changed_params) > 0,
                "call_count": 1,
            },
        ]

        action = config.spec.action
        count = len(sample_ids) * action.horizon * action.dimension
        shape = [len(sample_ids), action.horizon, action.dimension]
        action_evidence = {
            "semantics": config.spec.to_dict()["action"], "shape": shape,
            "attention_mask": [True] * count,
            "padding_mask": [False] * count,
            "loss_mask": [True] * count,
        }
        if action.representation == "continuous":
            action_evidence["actions"] = [targets[i % len(targets)] for i in range(count)]
            action_evidence["predictions"] = [preds[i % len(preds)] for i in range(count)]
        else:
            payload_ids = [x for x in range(action.vocabulary_size)
                           if x not in {action.bos_token_id, action.eos_token_id, action.pad_token_id}]
            canonical = [payload_ids[int(abs(targets[i % len(targets)]) * 100 + i) % len(payload_ids)] for i in range(count)]
            length = action.horizon * action.dimension
            for start in range(0, count, length):
                canonical[start] = action.bos_token_id
                canonical[start + length - 1] = action.eos_token_id
            coordinates = action.coordinates(len(sample_ids))
            tokens = [canonical[(b * action.horizon + h) * action.dimension + d] for b, h, d in coordinates]
            labels = action.shifted_labels(tokens)
            action_evidence.update(
                tokens=tokens, labels=labels, canonical_tokens=canonical, token_coordinates=coordinates,
                padding_mask=[x == action.pad_token_id for x in tokens],
                attention_mask=[x != action.pad_token_id for x in tokens],
                loss_mask=[x != action.pad_token_id and y != action.pad_token_id for x, y in zip(tokens, labels)],
            )
        if config.inject == "swap_action_tokens":
            key = "tokens" if action.representation == "discrete_tokens" else "actions"
            action_evidence[key] = list(reversed(action_evidence[key]))
        if config.inject == "swap_action_masks":
            action_evidence["loss_mask"][0] = False

        return {
            "schema_version": 1,
            "anatomy_spec_sha256": config.spec.sha256(),
            "action_evidence": action_evidence,
            "adapter_id": self.adapter_id,
            "case_id": config.case_id,
            "execution_environment": self.execution_environment,
            "evidence_scope": self.evidence_scope,
            "pid": os.getpid(),
            "seed": config.seed,
            "world_size": config.world_size,
            "inject": config.inject,
            "precision": {
                "dtype": config.dtype,
                "math_mode": "float32",
                "backend": "fake",
                "jit": False,
            },
            "sample_ids": sample_ids,
            "cameras": cameras,
            "batch_digest": batch_digest,
            "rng_initial_state": {
                "seed": config.seed,
                "init_seed": config.seed * 1009 + 7,
                "step_seed": config.seed * 7919 + 13,
            },
            "rng_events": rng_events,
            "step_before": 0,
            "step_after": 1,
            "forward_call_count": forward_call_count,
            "loss": loss,
            "grad_global_norm": grad_global_norm,
            "gradients": gradients,
            "weights_before": weights_before,
            "weights_before_digest": weights_before_digest,
            "weights_after": weights_after,
            "weights_after_digest": weights_after_digest,
            "changed_parameters": changed_params,
            "optimizer_applied": optimizer_applied,
            "rank_receipts": rank_receipts,
            "phases": phases,
        }


def main(argv: list[str] | None = None) -> int:
    """Subprocess entry point invoked by `capture.py` so each attempt runs in a fresh PID."""
    parser = argparse.ArgumentParser(
        description="Execute one deterministic FakeStepAdapter step in an isolated subprocess."
    )
    parser.add_argument("--output", required=True, help="Output JSON capture file path.")
    parser.add_argument("--spec-file", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--world-size", type=int, default=1)
    parser.add_argument("--case-id", default="pi05-cpu-mock-step-1")
    parser.add_argument("--sample-ids", default=",".join(DEFAULT_SAMPLE_IDS))
    parser.add_argument("--inject", default=None, choices=FAKE_NEGATIVE_INJECTIONS)
    args = parser.parse_args(argv)

    samples = tuple(s.strip() for s in args.sample_ids.split(",") if s.strip())
    spec = PI05_SPEC
    if args.spec_file:
        spec = load_anatomy_spec(args.spec_file)
    cfg = FakeStepConfig(
        spec=spec,
        seed=args.seed,
        sample_ids=samples,
        world_size=args.world_size,
        inject=args.inject,
        case_id=args.case_id,
    )
    payload = FakeStepAdapter().execute_step(cfg)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
