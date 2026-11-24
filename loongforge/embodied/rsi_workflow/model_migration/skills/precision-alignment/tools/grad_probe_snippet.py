#!/usr/bin/env python3
"""GRAD_PROBE — per-submodule LOCAL (pre-reduce) gradient norm (Rule 3).

PASTE-IN SNIPPET, not a runnable script. Add these three methods to the trainer on
both sides, call `_grad_probe_begin(raw)` before backward and
`_grad_probe_end(raw, phase)` after, then compare the `[GRAD_PROBE]` lines.

Why hooks and not `p.grad`: once a gradient-sharding optimizer owns the buffers
(DeepSpeed ZeRO's flat partition buffer, torch `ZeroRedundancyOptimizer`, or DDP
bucket views), `p.grad` after backward is a VIEW into a flat buffer and grads
outside this rank's shard can be stale — so a host-side read compares garbage.
`Tensor.register_hook` fires during autograd the instant the local grad is
produced, BEFORE any reduce/shard/clip mutates it. That gives the pure per-rank
local gradient, which is what isolates a backward difference, and rank0 sees the
same data shard on both sides so it is directly comparable.

🔑 The `n=` count is the point, not just the norm. Identical `n=825/292/241` on
both sides proves the trainable-parameter set and hook coverage match. Without it
a "matching" grad norm can be a coincidence over two different parameter subsets,
or a whole frozen/unfrozen module can silently differ. Always compare `n` first.

Expected precision: raw pre-reduce grad norms jitter roughly an order of magnitude
MORE than losses (measured on a bf16 flash-attn stack: loss ~1e-5 relative
run-to-run, grad norm ~2e-4 relative *within one side*). Size the band per
quantity — a 3e-4 grad-norm delta can be in-band while a 3e-4 loss delta is a bug.
Measure both sides' spread over >=3 runs before judging.

Default-inert: does nothing unless GRAD_PROBE=1. Costs ~0.15 s/step when on, so
turn it OFF for any performance comparison.
"""

SNIPPET = r'''
    def _init_grad_probe_hooks(self, raw) -> None:
        """Register per-parameter autograd hooks accumulating the LOCAL grad norm."""
        if os.environ.get("GRAD_PROBE") != "1":
            return
        if getattr(self, "_grad_probe_hooked", False):
            return
        self._grad_probe_hooked = True
        self._grad_probe_active = False
        self._grad_probe_acc, self._grad_probe_cnt = {}, {}

        def _make_hook(key):
            def _hook(grad):
                if not self._grad_probe_active or grad is None:
                    return
                self._grad_probe_acc[key] = (
                    self._grad_probe_acc.get(key, 0.0)
                    + float(grad.detach().float().pow(2).sum())
                )
                self._grad_probe_cnt[key] = self._grad_probe_cnt.get(key, 0) + 1
            return _hook

        # EDIT-ME: (bucket label, attribute name) per submodule you want split out.
        for key, modname in (("video", "video_model"),
                             ("action", "action_expert"),
                             ("und", "und_expert")):
            mod = getattr(raw, modname, None)
            if mod is None:
                continue
            for p in mod.parameters():
                if p.requires_grad:
                    p.register_hook(_make_hook(key))

    def _grad_probe_begin(self, raw) -> None:
        """Arm the hooks for the first few steps and reset the accumulators."""
        if os.environ.get("GRAD_PROBE") != "1":
            return
        self._init_grad_probe_hooks(raw)
        _gs = getattr(raw, "_parity_fm_step", None)
        self._grad_probe_active = (_gs is None or _gs <= 5)
        self._grad_probe_acc, self._grad_probe_cnt = {}, {}

    def _grad_probe_end(self, raw, phase: str) -> None:
        """After backward: print per-submodule local grad norm on rank0."""
        if os.environ.get("GRAD_PROBE") != "1" or not getattr(self, "_grad_probe_active", False):
            return
        self._grad_probe_active = False
        import math
        import torch.distributed as _dg
        if (_dg.get_rank() if _dg.is_available() and _dg.is_initialized() else 0) != 0:
            return

        def _n(key):
            return (math.sqrt(self._grad_probe_acc.get(key, 0.0)),
                    self._grad_probe_cnt.get(key, 0))

        vN, vn = _n("video")
        aN, an = _n("action")
        uN, un = _n("und")
        print(f"[GRAD_PROBE] phase={phase} fm_step={getattr(raw, '_parity_fm_step', None)} "
              f"video_gnorm={vN:.6e}(n={vn}) action_gnorm={aN:.6e}(n={an}) "
              f"und_gnorm={uN:.6e}(n={un})", flush=True)
'''

if __name__ == "__main__":
    print(__doc__)
    print(SNIPPET)
