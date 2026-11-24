#!/usr/bin/env python3
"""Effective DeepSpeed-config dump + field-by-field diff (BASE vs TRANSPLANT).

Rule 3 (backward/optimizer alignment) tool. The two sides often build the SAME
ds engine from the SAME json, but batch-size fields are auto-injected at
runtime (train_micro_batch_size_per_gpu / gradient_accumulation_steps /
train_batch_size = micro*GA*ws). This script constructs the *effective* config
each side actually runs and diffs it key-by-key, so a silent drift (e.g.
gradient_clipping present on one side only) is caught before it masquerades as
a numerical divergence.

REUSE: this is a reference template from the Motus port. Edit the four EDIT-ME
knobs for your project:
  - BASE_JSON / TRANS_JSON : the two ds config paths
  - WORLD / MICRO / GAS    : the launch geometry (from the run scripts)
  - the TRANSPLANT effective-config construction should mimic however your
    trainer injects the batch fields (here: motus_trainer._load_deepcompile_config)
  - the BASE effective-config construction should mimic however base builds its
    engine (here: accelerate DeepSpeedPlugin(hf_ds_config=...) -> auto fields)
"""
import json, copy

# ===== EDIT-ME: the two ds config paths =====
BASE_JSON = "/path/to/BASE/zero1.json"
TRANS_JSON = "/path/to/TRANSPLANT/zero1.json"

# ===== EDIT-ME: launch geometry (both sides must match) =====
WORLD = 8
MICRO = 8
GAS = 1

# ===== TRANSPLANT effective: mimic your trainer's config loader =====
with open(TRANS_JSON) as f:
    trans = json.load(f)
trans["train_micro_batch_size_per_gpu"] = MICRO
trans["gradient_accumulation_steps"] = GAS
trans["train_batch_size"] = MICRO * GAS * WORLD

# ===== BASE effective: construct the SAME engine base builds =====
# Here base wraps via accelerate DeepSpeedPlugin; adapt if base builds the
# engine a different way (direct deepspeed.initialize, etc).
base_plugin_cfg = None
base_err = None
try:
    import os
    os.environ.setdefault("ACCELERATE_USE_DEEPSPEED", "true")
    from accelerate.utils import DeepSpeedPlugin
    plugin = DeepSpeedPlugin(hf_ds_config=BASE_JSON)
    hf = getattr(plugin, "hf_ds_config", None)
    if hf is not None and hasattr(hf, "config"):
        base_plugin_cfg = copy.deepcopy(hf.config)
    else:
        base_plugin_cfg = copy.deepcopy(getattr(plugin, "deepspeed_config", {}))
    for k, v in {
        "train_micro_batch_size_per_gpu": MICRO,
        "gradient_accumulation_steps": GAS,
        "train_batch_size": MICRO * GAS * WORLD,
    }.items():
        if base_plugin_cfg.get(k, "auto") == "auto":
            base_plugin_cfg[k] = v
    import accelerate
    acc_ver = accelerate.__version__
except Exception as e:
    base_err = repr(e)
    acc_ver = "N/A"

print("accelerate version:", acc_ver)
print("\n===== TRANSPLANT effective ds config =====")
print(json.dumps(trans, indent=2, sort_keys=True))
print("\n===== BASE effective ds config =====")
if base_err:
    print("ERROR building base config:", base_err)
else:
    print(json.dumps(base_plugin_cfg, indent=2, sort_keys=True))

# ===== field-by-field diff =====
if not base_err:
    print("\n===== FIELD DIFF (base vs trans) =====")
    keys = sorted(set(trans) | set(base_plugin_cfg))
    any_diff = False
    for k in keys:
        b = base_plugin_cfg.get(k, "<MISSING>")
        t = trans.get(k, "<MISSING>")
        mark = "OK " if b == t else ">>>"
        if b != t:
            any_diff = True
        print(f"{mark} {k}: base={b!r}  trans={t!r}")
    print("\nRESULT:", "DIFFERENCES FOUND" if any_diff else "ALL FIELDS MATCH")
