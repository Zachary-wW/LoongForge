---
name: loongforge-debug
description: Use when encountering training errors, hangs, OOM, loss anomalies, or checkpoint issues in LoongForge. Provides structured debugging workflows with domain-specific decision trees. Triggers on 'training error', 'NCCL timeout', 'OOM', 'loss NaN', 'loss spike', 'checkpoint mismatch', 'hang', 'XPU error', 'debug', 'crash', 'broken'.
---

# LoongForge Training Debugging

## Triage

**Quick Path** — error message is clear, matches a known pattern below:
1. Match symptom in the Known Failure Patterns table
2. Run the diagnostic steps
3. Apply the fix
4. Verify with a short run

**Full Protocol** — cause is unclear, intermittent, or multi-node:
1. Classify failure mode: crash / hang / correctness / performance
2. Isolate scope: single rank? all ranks? specific training step?
3. Reproduce at minimal scale (1-2 GPUs, small model, few steps)
4. Apply domain-specific checklist below
5. Fix, verify, document

## Known Failure Patterns

| Symptom | Likely Cause | Diagnostic | Fix |
|---|---|---|---|
| `NCCL timeout` / `watchdog timeout` | Rank divergence or network issue | Set `NCCL_DEBUG=INFO`; check if all ranks reach the same collective; check data loader balance | Fix dp_balance config; verify TP/PP/DP topology matches `--nproc_per_node * --nnodes` |
| `CUDA OOM` | Batch too large or activation memory | `nvidia-smi` per rank; check micro_batch_size vs model size | Reduce `--micro-batch-size`; enable `--recompute-granularity selective`; reduce `--seq-length` |
| `Loss NaN` at step 0 | Checkpoint load mismatch or missing weights | Check for "missing keys" / "unexpected keys" in load log; verify key_mappings | Fix convert YAML name_map; verify TP resharding dimensions |
| `Loss NaN` after N steps | LR too high or gradient explosion | Check grad norm in logs; look for sudden spike before NaN | Lower `--lr`; set `--clip-grad 1.0`; check data for corrupt samples |
| `Loss spike` (recovers) | Bad data batch or FP8 instability | Log batch index; inspect data at that offset | Skip corrupt samples; adjust FP8 dynamic scaling policy |
| `Checkpoint key mismatch` | key_mappings out of sync with model code | Diff expected vs actual keys from error message | Update `tools/convert_checkpoint/key_mappings/`; re-run conversion |
| `Checkpoint shape mismatch` | TP/PP split dimension wrong | Check source TP vs target TP; check which dim is split | Fix TP dimension in `module_convertor/model.py` (column-parallel: split output_size; row-parallel: split input_size) |
| `Hang (no error, no progress)` | PP deadlock or data loader starvation | `py-spy dump --pid <rank0>`; check if stuck in NCCL wait or data I/O | Fix PP schedule in `model_chunk_schedule_plan.py`; increase data workers |
| `XPU: op not implemented` | Missing XPU dispatch path | Check `dispatch.py` MultiAccModules for the op | Add XPU fallback path or use CPU implementation |
| `ImportError` from Megatron | Submodule version mismatch | Check `third_party/Loong-Megatron` commit hash | `git submodule update --init`; verify patches/ apply cleanly |
| `AssertionError` in trainer_builder | Model family not in dispatch set | Check constants.py has the family; check train/__init__.py imports | Add family to constants.py; import trainer module in train/__init__.py |

## Domain Checklists

### Checkpoint Conversion

- Compare `model.state_dict().keys()` between HF and mcore formats
- Verify TP dimension: column-parallel layers split on `output_size`, row-parallel on `input_size`
- For MoE: check expert routing keys in `key_reverser_expert.py`
- For VLM: verify all 3 components converted (language + encoder + projector)
- Run with `--dry-run` first to see key mapping without writing files

### Distributed Training

- Verify: world_size == TP × PP × DP × EP
- Check `CUDA_VISIBLE_DEVICES` assignment per rank
- Verify `--tensor-model-parallel-size` and `--pipeline-model-parallel-size` match intended config
- For MoE: check `--expert-model-parallel-size` and `--num-experts` relationship
- For CP (context parallel): verify sequence length is divisible by CP size

### VLM-Specific

- Verify encoder output shape matches projector input expectation
- Check `mm_plugin.py` returns correct modality token count
- Verify `omni_combination_model.py` chunk schedule for PP
- Check image/video preprocessing dimensions in `data/multimodal/`

### Data Pipeline

- Check `dp_balance/` for sequence length distribution across ranks
- Verify `chat_template.py` produces correct token IDs for the model's tokenizer
- Check padding/truncation in sft_data_collator.py
- For multimodal: verify image resolution and patch count match encoder config

## Three-Strike Rule

After 3 consecutive failed fix attempts:
1. Stop fixing symptoms
2. Question whether the underlying approach is wrong
3. Re-read the error from scratch with fresh eyes
4. Report findings to the user before continuing

## Escalation

Stop and ask for human help when:
- Intermittent failures that cannot be reproduced at minimal scale
- Suspected hardware issues (ECC errors, NVLink failures, GPU throttling)
- Loong-Megatron internal bugs requiring upstream submodule fix
- TransformerEngine patch incompatibility requiring new patch development
