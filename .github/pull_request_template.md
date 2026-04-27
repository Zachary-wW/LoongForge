## What does this PR do?
<!-- One-line description of the change. Reference related issues: closes #xxx -->

## Changes
-

## Type
- [ ] `feat` — new model / feature
- [ ] `fix` — bug fix
- [ ] `perf` — performance improvement
- [ ] `refactor` — code cleanup, no behavior change
- [ ] `test` — tests / CI
- [ ] `docs` — documentation
- [ ] `chore` — build, deps, infra

## Checklist
- [ ] Pre-commit checks pass (`ruff check --fix && ruff format`)
- [ ] If new model: `constants.py` updated, config YAML added, `config_map.py` updated, example scripts added under `examples/<model>/`
- [ ] If data format or packing logic changed: downstream dataset compatibility verified
- [ ] If checkpoint format changed: backward compatibility considered
- [ ] If new dependency: added to `pyproject.toml` with pinned version

## Validation
<!-- E2E test results, loss curves, or manual verification steps. Leave blank if fully covered by CI. -->
