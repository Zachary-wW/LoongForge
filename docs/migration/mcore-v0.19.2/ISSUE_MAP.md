# Issue and Copy Inventory Map

This map connects the existing fork Issues to stable copy-inventory IDs. The
Issue remains the unit of scope and decision-making; the inventory provides
file-level evidence.

| Issue | Primary scope | Copy inventory IDs | Notes |
|---|---|---|---|
| #34 | Training runtime | `106-121` | arguments, checkpointing, trainer, parser, pretrain/SFT/loss and training utils |
| #35 | Wan CP and diffusion interfaces | `024-032`, `108-109` | Wan/Qwen-Image pretrain entries plus attention/layer/model/block/provider |
| #36 | Obsolete validation | `122` | validators |
| #37 | Shared utility, data and MTP APIs | `001-016`, `102-103`, `118` | data builders, rerun iterator, common transformer helpers, schedule and shared utils |
| #38 | LLM/VLM model construction APIs | Model-side entries outside DSA/CSA | Foundation and encoder models, attention, experts/router and layer specs |
| #39 | DSA/CSA and experimental attention | `010-014`, `060-064` | DSA/CSA vendor files and DeepSeek experimental attention |
| #40 | Upstream duplicate-code disposition | cross-cutting | Records which IDs become `reuse upstream` or `keep LoongForge fork` |
| #41 | Integration and validation | all 123 IDs | Assigns test evidence to an integration SHA |

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
