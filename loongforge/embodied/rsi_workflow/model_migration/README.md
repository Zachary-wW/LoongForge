# Model Migration

The workspace for migrating open-source models into LoongForge: while keeping the logic consistent and the precision aligned, adapt the open-source model's format and structure to LoongForge's model conventions so that it can be directly trained, evaluated, and released inside the LoongForge framework.

The migration work is organized around the RSI (Robust Self-Improvement / Recurrent Self-Improvement) agent workflow: the scheduling and execution logic in `core` drives the process, the skills in `skills` are reused to carry out concrete operations, the outputs are archived as reproducible tasks in `examples` after passing the checks in `validate`, and the whole process is documented in `docs`.

## Directory Structure

| Directory | Description |
| --- | --- |
| [`core/`](core/README.md) | Core code. The main part of the migration work, covering the RSI-related implementations such as sandbox, templates, and scheduling/control |
| [`docs/`](docs/README.md) | Documentation. Usage guides and samples for the features |
| [`examples/`](examples/README.md) | Reference tasks. Migration tasks that have already been run end-to-end and can be reused as templates |
| [`skills/`](skills/README.md) | Skill collection. The skills mainly used during migration, collected here and invoked by `core` |
| [`validate/`](validate/README.md) | Validation module. Checks the accuracy of the final output, including logic consistency and precision alignment, to make sure the result is correct and usable |

## Workflow

1. Pick or create a migration task in `examples/` as the target.
2. `core/` schedules the skills in `skills/` to perform the model migration and format conversion according to the task description.
3. Submit the migration output to `validate/` for logic and precision checks.
4. After validation passes, archive the process as a reproducible task in `examples/` and document the usage in `docs/`.
