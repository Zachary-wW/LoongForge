# Validate

The validation-module directory of the model-migration workspace. It is responsible for checking the accuracy of the final output, including logic consistency and precision alignment, to make sure the migration result is correct and usable — the last gate before a migration output is merged.

## Responsibilities

- **Logic consistency check**: verifies that the migrated model matches the original open-source model in structure, data flow, and inference path.
- **Precision alignment check**: compares the outputs of the pre- and post-migration models under identical inputs and weights, confirming the numerical precision is aligned (within the allowed tolerance).
- **Usability check**: confirms the migration output can be trained, evaluated, and released inside LoongForge with reproducible results.

## Contents

- Validation scripts: implementations of the logic comparison, precision comparison, and smoke tests.
- Validation configuration: check items and thresholds for different models/tasks.
- Validation reports: results and conclusions of each validation run, used as the merge basis.
