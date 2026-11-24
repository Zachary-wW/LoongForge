#!/usr/bin/env python3
"""VPROBE2 — layer-by-layer activation hashing to localize WHERE a forward first
diverges (Rule 2).

PASTE-IN SNIPPET, not a runnable script. Add `_vprobe_dump` to both sides' model
file, then sprinkle call sites down the forward. Run 1 step per side and compare
the `[VPROBE2]` lines: the first tag whose hash differs is the first divergent op.

This is the tool that turns "the video loss differs by 1e-4" into "layer 4's video
branch is the first thing that differs, and layers 0-3 plus the same layer's
action/und branches are bit-identical" — which immediately rules out data, weights
and any global config, and points at one op.

Placement strategy (bisect, do not instrument everything at once):
  1. Coarse first: one call per LOSS INPUT (`video_pred`, `action_pred`, targets).
     If a target matches but the pred does not, the divergence is upstream of the
     loss, so stop auditing loss precision.
  2. Then per-layer at the block boundary: `_vprobe_dump(f"L{i}", v=v, a=a, u=u)`.
     Hash every modality at the same point — "L4 video differs, L4 action/und
     identical" is a much stronger clue than "L4 differs".
  3. Then intra-block on the one layer that first differs: adaln / attn / cross /
     ffn.
  Use VPROBE2_TAGS to re-run only the narrowed set instead of editing code again.

Env knobs (all default-inert; unset VPROBE2 == zero overhead, and the call sites
can stay in the tree permanently):
  VPROBE2=1              enable
  VPROBE2_MAX=64         cap total emitting calls, so only step-1 gets dumped
  VPROBE2_TAGS=L4,L5     comma-separated tag PREFIXES to narrow to
  VPROBE2_DIR=/tmp/vp    also .save() the fp32 tensors, so when hashes differ you
                         can measure the MAGNITUDE of the delta (1 ULP vs real
                         bug) instead of only knowing "different"

⚠️ Two traps:
  - Hashes must be taken in fp32 (`.float()`) on rank0 only, or you are comparing
    different data shards and everything will "differ".
  - These are side-effecting host reads. Under `torch.compile` they force a graph
    break; guard with `torch.compiler.is_compiling()` if the region is compiled,
    and do the bisect in EAGER mode.
"""

SNIPPET = r'''
_VPROBE2_CALLS = 0


def _vprobe_dump(tag: str, **tensors) -> None:
    """VPROBE2 (default unset -> inert): rank-0 fp32 sha1 of named intermediates."""
    if os.environ.get("VPROBE2") is None:
        return
    _sel = os.environ.get("VPROBE2_TAGS")
    if _sel and not any(tag.startswith(p) for p in _sel.split(",")):
        return
    import hashlib as _h
    import torch.distributed as _d
    if (_d.get_rank() if _d.is_available() and _d.is_initialized() else 0) != 0:
        return
    global _VPROBE2_CALLS
    if _VPROBE2_CALLS >= int(os.environ.get("VPROBE2_MAX", "64")):
        return
    _VPROBE2_CALLS += 1
    _dir = os.environ.get("VPROBE2_DIR")
    parts = []
    for k, v in tensors.items():
        if isinstance(v, (list, tuple)) and v and torch.is_tensor(v[0]):
            v = torch.stack([u.float() for u in v], dim=0)
        if not torch.is_tensor(v):
            parts.append(f"{k}=None")
            continue
        f = v.detach().float().cpu().contiguous()
        parts.append(f"{k}={tuple(v.shape)}:{_h.sha1(f.numpy().tobytes()).hexdigest()[:12]}")
        if _dir:
            os.makedirs(_dir, exist_ok=True)
            torch.save(f, os.path.join(_dir, f"{tag}.{k}.pt"))
    print(f"[VPROBE2] {tag} " + " ".join(parts), flush=True)


# call sites, e.g. inside the per-layer loop:
#     _vprobe_dump(f"L{i}", v=video_h, a=action_h, u=und_h)
'''

if __name__ == "__main__":
    print(__doc__)
    print(SNIPPET)
