"""OpenPI JAX (`pi05_libero`) single-step adapter skeleton & preflight checks.

Fail-closed policy:
- Never downloads checkpoints, LeRobot datasets, tokenizers, or norm stats.
- Never fakes a `pass` when required local assets, JAX runtime, or GPU are unavailable;
  explicitly emits `not_run` + `WAITING_ASSET` / `RUNTIME_NOT_AVAILABLE`.
"""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, TypeAlias

PreflightStatus: TypeAlias = Literal["ready", "waiting", "not_run", "blocked"]


@dataclass(frozen=True, slots=True)
class OpenPIJaxConfig:
    """Pinned local inputs required to execute a real OpenPI JAX single step."""

    upstream_repo: Path
    checkpoint_path: Path | None = None
    dataset_path: Path | None = None
    tokenizer_path: Path | None = None
    norm_stats_path: Path | None = None
    config_name: str = "pi05_libero"
    seed: int = 42
    batch_size: int = 2
    require_gpu: bool = True


@dataclass(frozen=True, slots=True)
class OpenPIPreflightResult:
    """Structured preflight report for real OpenPI JAX single-step capture."""

    status: PreflightStatus
    execution_environment: str
    reason_codes: tuple[str, ...]
    missing_assets: tuple[str, ...]
    summary: str
    runtime_available: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "execution_environment": self.execution_environment,
            "reason_codes": list(self.reason_codes),
            "missing_assets": list(self.missing_assets),
            "summary": self.summary,
            "runtime_available": self.runtime_available,
        }


class OpenPIJaxAdapter:
    """Real-model adapter skeleton for OpenPI JAX (Batch 4 hardware smoke)."""

    adapter_id = "openpi_jax_step_adapter"
    execution_environment = "openpi_jax"
    evidence_scope = "real_single_step"

    def preflight(self, config: OpenPIJaxConfig) -> OpenPIPreflightResult:
        """Verify local assets and runtime availability without downloading anything."""
        missing_assets: list[str] = []
        reason_codes: list[str] = []

        def _add(code: str) -> None:
            if code not in reason_codes:
                reason_codes.append(code)

        if not config.upstream_repo.is_dir():
            missing_assets.append("upstream_repo")
            _add("WAITING_ASSET")

        for label, path in (
            ("checkpoint", config.checkpoint_path),
            ("dataset", config.dataset_path),
            ("tokenizer", config.tokenizer_path),
            ("norm_stats", config.norm_stats_path),
        ):
            if path is None or not Path(path).exists():
                missing_assets.append(label)
                _add("WAITING_ASSET")

        jax_installed = importlib.util.find_spec("jax") is not None
        if not jax_installed:
            _add("JAX_RUNTIME_MISSING")

        gpu_ready = False
        if jax_installed and config.require_gpu:
            try:
                import jax  # type: ignore[import-not-found]

                gpu_ready = any(dev.platform == "gpu" for dev in jax.devices())
            except Exception:
                gpu_ready = False
            if not gpu_ready:
                _add("GPU_NOT_AVAILABLE")

        runtime_available = jax_installed and (gpu_ready or not config.require_gpu)
        if missing_assets or not runtime_available:
            return OpenPIPreflightResult(
                status="not_run",
                execution_environment=self.execution_environment,
                reason_codes=tuple(reason_codes),
                missing_assets=tuple(missing_assets),
                summary=(
                    "OpenPI JAX single-step capture not_run: missing local assets or GPU runtime "
                    f"(missing={missing_assets}, runtime_available={runtime_available})"
                ),
                runtime_available=runtime_available,
            )

        return OpenPIPreflightResult(
            status="ready",
            execution_environment=self.execution_environment,
            reason_codes=(),
            missing_assets=(),
            summary="OpenPI JAX local assets and runtime preflight passed",
            runtime_available=True,
        )

    def execute_step(self, config: OpenPIJaxConfig) -> dict[str, Any]:
        """Execute one real OpenPI JAX step once assets & GPU are available (Batch 4)."""
        pre = self.preflight(config)
        if pre.status != "ready":
            return {
                "schema_version": 1,
                "adapter_id": self.adapter_id,
                "execution_environment": self.execution_environment,
                "evidence_scope": self.evidence_scope,
                "status": "not_run",
                "reason_codes": list(pre.reason_codes),
                "missing_assets": list(pre.missing_assets),
                "summary": pre.summary,
            }
        raise NotImplementedError(
            "OpenPIJaxAdapter real GPU single-step execution is scheduled for Batch 4 "
            "once pi0.5 + LIBERO assets and GPU quota are provisioned"
        )
