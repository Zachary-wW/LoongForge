#!/usr/bin/env python3
"""WPROBE — one-shot weight dump, for proving BASE and TRANSPLANT hold the SAME
weights before you blame an operator (Rule 2).

This file is a PASTE-IN SNIPPET, not a runnable script. Drop the block below at
the top of the model's training entry point (`training_step` / `forward`) on BOTH
sides, run 1 step each, then diff the two dumps with `diff_weight_probe.py`.

When to reach for it: step-1 forward differs, inputs are already proven identical
by the dataloader checksum. Either a weight differs (checkpoint load path, a dtype
cast, an uninitialized head, a different init seed for a non-pretrained module) or
an op differs. This names the tensor in one run per side, and its `dtype` column
doubles as the parameter-storage-dtype probe that
`model-adaptation/rules/03-trainer-alignment.md` asks for.

Output format (one line per tensor, `P`=parameter, `B`=buffer):
    P video_model.blocks.0.ffn.0.weight (5120, 1024) torch.bfloat16 3f2a9c1e0b7d4a55

Default-inert: does nothing unless $WPROBE names an output path.
    BASE : WPROBE=/tmp/wprobe_base.txt  <base launch cmd>
    TRANS: WPROBE=/tmp/wprobe_trans.txt <trans launch cmd>
    python diff_weight_probe.py /tmp/wprobe_base.txt /tmp/wprobe_trans.txt
"""

SNIPPET = r'''
# --- WPROBE (default unset -> inert) -------------------------------------
# One-shot rank0 dump of every parameter/buffer's fp32 sha1 to $WPROBE.
if os.environ.get("WPROBE") is not None and not getattr(self, "_wprobe_done", False):
    self._wprobe_done = True
    import hashlib as _hw
    import torch.distributed as _dw
    if (_dw.get_rank() if _dw.is_available() and _dw.is_initialized() else 0) == 0:
        def _wsha(t):
            t = t.detach()
            # view_as_real: complex rope/freq buffers cannot go through .float()
            t = torch.view_as_real(t) if t.is_complex() else t.float()
            return _hw.sha1(t.cpu().contiguous().numpy().tobytes()).hexdigest()[:16]
        with open(os.environ["WPROBE"], "w") as _f:
            for _n, _p in self.named_parameters():
                _f.write(f"P {_n} {tuple(_p.shape)} {_p.dtype} {_wsha(_p)}\n")
            for _n, _bf in self.named_buffers():
                _f.write(f"B {_n} {tuple(_bf.shape)} {_bf.dtype} {_wsha(_bf)}\n")
        print(f"[WPROBE] wrote {os.environ['WPROBE']}", flush=True)
# ------------------------------------------------------------------------
'''

if __name__ == "__main__":
    print(__doc__)
    print(SNIPPET)
