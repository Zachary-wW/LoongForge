# Skills

The skill-collection directory of the model-migration workspace. The skills (agent capabilities) mainly used during migration are gathered here and invoked by `core`, so that capabilities do not scatter across the workspace.

## Responsibilities

- Centrally manages the skills the migration workflow depends on, acting as the capability source for `core`.
- Each skill encapsulates a category of reusable operations (e.g., code generation, format conversion, command execution, check-and-fix), invoked on demand by the scheduling flow.
- Keeps skills decoupled from `core`: to add a capability, just add the corresponding skill under `skills` without touching `core`.

## Contents

- Skill definitions: each subdirectory or file corresponds to a skill, including its capability description, interface, and usage constraints.
- Skill index: describes the purpose and applicable scenarios of each skill for scheduling selection.
