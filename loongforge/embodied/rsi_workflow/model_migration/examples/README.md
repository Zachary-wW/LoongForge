# Examples

The reference-task directory of the model-migration workspace. It holds open-source model migration tasks that have already been run end-to-end, serving as templates and validation baselines for later migrations.

## Responsibilities

- Captures migration tasks that have already been run through, forming reusable reference samples.
- Each task contains the complete migration input, execution process, and final output, and can be reproduced directly against it.
- Serves as the starting point for new migration tasks: prefer the patterns of existing tasks instead of starting from scratch.

## Contents

- Migration tasks: each subdirectory corresponds to a migrated open-source model, recording its data, configuration, scripts, and outputs.
- Reproduction notes: the environment, commands, and expected results required to run a task.
