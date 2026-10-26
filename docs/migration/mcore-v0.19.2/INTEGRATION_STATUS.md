# Loong-Megatron v0.19.2 Integration Status

This file tracks the prerequisite for updating LoongForge's
`third_party/Loong-Megatron` submodule. It intentionally leaves the current
submodule pointer unchanged until a complete integration commit is available.

## Current facts

- LoongForge `master` points at
  `b10469fc8b25c9cf805cacf5c3c91121faf123b3` on
  `loong-main/core_v0.15.0`.
- The Loong-Megatron v0.19.2 target base is
  `4b4acac9a1d28ea6829c8d4f566d75698a21249d`.
- Loong-Megatron contains 38 separate OPEN Draft migration PRs.
- The fork's `review/mcore-v0.19.2` branch currently resolves to
  `9f9befd0e75042096bfcaed3165d0ef03e2c7a4f`.
- That review branch is only one commit ahead of the target base and must not
  be treated as the aggregate v0.19.2 integration head.

## Required integration head

The submodule update PR is ready to proceed only when a shared, fetchable
Loong-Megatron branch contains the reviewed result of the intended migration
PR set. The integration head must:

1. Be available from a URL that CI and other developers can fetch.
2. Include the reviewed v0.19.2 migration commits, not only a single feature.
3. Pass import, symbol, CLI and constructor checks.
4. Be pinned by full commit SHA in the LoongForge submodule PR.

## Deferred validation

The following checks run after the integration head is fixed:

- Full changed-module import audit.
- CLI and model construction smoke.
- Representative dense and MoE training steps.
- FP8/HDO/offload and checkpoint save/resume.
- VLM and Diffusion smoke.
- Comprehensive static and runtime validation across affected PRs.

Until these conditions are met, Issue #41 remains blocked and must not be
reported as completed.
