# F1/F2 Negative Cases

## F1

- Missing upstream commit.
- Dirty upstream worktree.
- Entrypoint deleted at the pinned commit.
- Paper value used as the runtime configuration.
- Missing weight or dataset still reported as PASS.
- Unknown model silently mapped to a pi05 template.
- Source tree changed during analysis.

## F2

- One forward call skipped.
- Camera order changed.
- Sample order changed.
- RNG draw order changed.
- Noise or timestep changed.
- Loss branch changed.
- Optimizer step skipped.
- Work executed only on rank 0.
- Eval-mode output used as a training step.
- Anatomy emitted without a frozen source commit or Validator.
- Segment view accepted as a substitute for behavior assertions.

Each case must produce a machine-readable failure reason and evidence path.
