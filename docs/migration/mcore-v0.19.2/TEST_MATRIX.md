# LoongForge Megatron v0.19.2 Test Matrix

This matrix links the existing migration Issues to shared test cases. It is a
planning artifact: a passing test is recorded against its commit SHA and then
linked from every affected Issue or PR.

## Test levels

| Level | Trigger | Purpose | Hardware |
|---|---|---|---|
| L0 | Every commit and PR update | Syntax, lint, import/API symbol checks, diff hygiene | CPU |
| L1 | Every completed migration unit | Targeted unit or constructor test | CPU or one accelerator |
| L2 | Each five-to-eight-PR integration wave | Short distributed smoke test | 2-8 accelerators |
| L3 | Completed feature unit | Feature-specific forward/backward and save/resume | Required target hardware |
| L4 | Frozen integration head | Representative model/loss/resume/performance matrix | GPU/XPU test pool |

## Shared test cases

| Test ID | Scenario | Issue scope | Copy inventory scope | Minimum steps | Pass criteria |
|---|---|---|---|---|---|
| `LF-T0-STATIC` | Full import and changed-symbol audit | #34-#41 | All 80 entries | 0 | No unresolved import or changed-signature failure |
| `LF-T1-TRAIN` | Trainer setup and dataset iterator construction | #34, #37 | `001`, `005-007`, `070-080` | 0 | Trainer and dataloader construct |
| `LF-T2-DENSE` | Dense model build and 2-4 training steps | #37, #38, #40 | Model/GPT/block/attention/layer-spec entries | 4 | Finite loss, no graph or shape error |
| `LF-T2-DIST` | Dense TP2/PP2/DP2 short run | #37, #38, #40 | GPT/block/attention/process-group entries | 4 | Same loss sanity as TP1, no collective mismatch |
| `LF-T3-MOE` | MoE model EP2 short run | #38, #39 | Router, experts, layer-spec, DSA entries | 4 | Finite loss, token counts conserved |
| `LF-T4-FP8` | FP8/HDO/offload checkpoint cycle | #34, #37, #41 | Checkpointing/optimizer/offload entries | 2+resume 1 | Save, reload and continue with finite loss |
| `LF-T5-VLM` | Representative VLM build and short run | #38, #40 | Encoder/model/layer-spec/rope entries | 2 | Image-text batch completes forward/backward |
| `LF-T6-DIFFUSION` | Wan CP2 batch broadcast and short run | #35, #41 | API-coupled Wan files | 2 | Both CP ranks consume correct broadcast |
| `LF-T7-CHECKPOINT` | Dense or MoE checkpoint roundtrip | #34, #37, #41 | Checkpoint/runtime entries | 2+resume 1 | Step, RNG, optimizer and scheduler restore |
| `LF-T8-INTEGRATION` | LoongForge + integrated Loong-Megatron smoke | #41 | All applicable entries | 4 | Import, model build, train step and cleanup succeed |

## Representative configurations

| Config ID | Model class | Parallelism | Notes |
|---|---|---|---|
| `C1-DENSE-1GPU` | Small dense GPT/LLM | TP1 PP1 DP1 | Primary preflight |
| `C2-DENSE-SMALL` | Small dense GPT/LLM | TP2 PP2 DP2 | Distributed APIs and process groups |
| `C3-MOE` | Small MoE | TP1 PP2 EP2 | Router, experts and dispatcher |
| `C4-FP8` | Small FP8/HDO model | Existing supported topology | Optimizer/offload/checkpoint |
| `C5-VLM` | Qwen-VL or InternVL | Smallest supported topology | ViT + language model |
| `C6-WAN-CP` | Wan diffusion | CP2 | CP broadcast and data path |

The exact model names, batch sizes and datasets will be frozen when the
integration head is selected. Until then these are test-shape contracts, not
claims that a run has passed.

## Result record

Every executed case must record:

- `test_id`
- Issue and PR numbers
- Copy inventory IDs
- Loong-Megatron base and integration SHAs
- LoongForge SHA
- Config, topology and hardware
- Steps completed
- Pass/fail and error signature
- Log or artifact location
- Last verified time

No row is marked passed until a concrete commit SHA and run artifact exist.
