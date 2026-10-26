# Issue and Copy Inventory Map

This map connects the existing fork Issues to stable copy-inventory IDs. The
Issue remains the unit of scope and decision-making; the inventory provides
file-level evidence.

| Issue | Primary scope | Copy inventory IDs | Notes |
|---|---|---|---|
| #34 | Training runtime | `LF-COPY-072` to `LF-COPY-078` | checkpointing, training utils, initialize, pretrain/SFT/loss |
| #35 | Wan CP source rank | none | Linked through the separate changed-API scan; Wan files are not in the confirmed copy inventory |
| #36 | Obsolete validation | `LF-COPY-079` | validators |
| #37 | Shared utility, batch and MTP APIs | `001`, `005-007`, `070-071`, `080` | rerun iterator, MLA/local layers, schedule helpers, shared utils |
| #38 | Model construction APIs | Model-specific entries in `008-069` | Excludes DSA/CSA entries assigned to #39 |
| #39 | DSA/CSA and experimental attention | `002-004`, `031-035` | Vendor and experimental attention paths |
| #40 | Upstream duplicate-code disposition | cross-cutting | Records which IDs become `reuse upstream` or `keep LoongForge fork` |
| #41 | Integration and validation | all applicable IDs | Assigns test evidence to an integration SHA |

## Decision values

Each Copy ID receives one of these review outcomes:

- `reuse-upstream`: delete the local implementation and import Megatron.
- `port-delta`: start from the target file and transplant only the LoongForge
  delta.
- `keep-fork`: keep an explicit LoongForge implementation with documented
  provenance and synchronization ownership.
- `api-only`: the file uses Megatron APIs but does not contain copied logic.
- `defer`: provenance or target behavior is not yet clear.

## Link requirements

- Issues link to this directory through a pinned commit permalink.
- Docs PRs list the affected Issues.
- Code PRs list the exact Copy IDs they change.
- Test results link the integration SHA and are reused by every affected PR.
