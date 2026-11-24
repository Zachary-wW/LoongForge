# Core

The core-code directory of the model-migration workspace, and the main part of the migration work. It concentrates the RSI (Robust Self-Improvement / Recurrent Self-Improvement) implementations, including the sandbox, templates, and scheduling/control.

## Responsibilities

- **Sandbox**: provides an isolated, controlled execution environment for model migration, format conversion, and other operations, preventing side effects of the migration from polluting the repository or the host environment.
- **Templates**: provides reusable code and configuration templates for wiring a model into LoongForge, so that migration outputs structurally follow LoongForge conventions.
- **Scheduling & Control**: the driving core of the RSI workflow. It is responsible for task decomposition, step scheduling, skill invocation, and state management, chaining `skills`, `validate`, and other modules into a complete migration loop.

## Contents

- Sandbox: environment creation, dependency injection, isolated execution, and cleanup.
- Templates: reusable templates for model configuration, model implementation, data integration, and launch scripts.
- Scheduling: task description parsing, step orchestration, skill dispatch, and result aggregation.
