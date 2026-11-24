#!/usr/bin/env python3
"""Diff two WPROBE weight dumps and report per-tensor mismatches (Rule 2).

Companion to weight_probe_snippet.py. Project-independent — no EDIT-ME.

    python diff_weight_probe.py /tmp/wprobe_base.txt /tmp/wprobe_trans.txt

Reads lines of the form
    P <name> <shape-tuple> <dtype> <sha1>
and reports, in priority order:
  1. name-set differences  — a module exists on one side only, or is named
     differently. Fix this FIRST; a missing tensor invalidates any grad-norm or
     parameter-count comparison.
  2. shape mismatches      — config drift (hidden size, layer count, vocab).
  3. dtype mismatches      — the parameter-storage-dtype question. A single
     bf16-vs-fp32 tensor changes the forward.
  4. value (sha1) mismatches — same name/shape/dtype, different numbers:
     checkpoint load path, an init seed for a non-pretrained head, or a cast.
Value mismatches are grouped by the first 4 name components so a whole subtree
("all 30 blocks of video_model") reads as one finding instead of 300 lines.

Exit code 0 = fully identical, 1 = any mismatch, so it can gate a script.
"""
import re
import sys
from collections import defaultdict

LINE = re.compile(r'^([PB]) (\S+) (\(.*?\)) (\S+) ([0-9a-f]+)$')


def load(path):
    d = {}
    for line in open(path):
        m = LINE.match(line.rstrip("\n"))
        if not m:
            if line.strip():
                print(f"UNPARSED: {line.rstrip()}", file=sys.stderr)
            continue
        kind, name, shape, dtype, sha = m.groups()
        d[(kind, name)] = (shape, dtype, sha)
    return d


def main():
    if len(sys.argv) != 3:
        sys.exit("usage: diff_weight_probe.py <base_dump> <trans_dump>")
    a, b = load(sys.argv[1]), load(sys.argv[2])

    only_a, only_b = sorted(set(a) - set(b)), sorted(set(b) - set(a))
    common = sorted(set(a) & set(b))
    print(f"base entries={len(a)} trans entries={len(b)} common={len(common)}")
    print(f"only-in-base={len(only_a)} only-in-trans={len(only_b)}")
    for k in only_a[:20]:
        print("  ONLY-BASE ", k)
    for k in only_b[:20]:
        print("  ONLY-TRANS", k)

    shape_diff, dtype_diff, sha_diff = [], [], []
    for k in common:
        (sa, da, ha), (sb, db, hb) = a[k], b[k]
        if sa != sb:
            shape_diff.append((k, sa, sb))
        elif da != db:
            dtype_diff.append((k, da, db))
        elif ha != hb:
            sha_diff.append((k, ha, hb))

    print(f"\nshape mismatches={len(shape_diff)} dtype mismatches={len(dtype_diff)} "
          f"value(sha) mismatches={len(sha_diff)}")
    for k, x, y in shape_diff[:40]:
        print(f"  SHAPE  {k[0]} {k[1]}  base={x} trans={y}")
    for k, x, y in dtype_diff[:40]:
        print(f"  DTYPE  {k[0]} {k[1]}  base={x} trans={y}")
    for k, x, y in sha_diff[:200]:
        print(f"  VALUE  {k[0]} {k[1]}  base={x} trans={y}")

    if sha_diff:
        groups = defaultdict(int)
        for k, _, _ in sha_diff:
            groups[".".join(k[1].split(".")[:4])] += 1
        print("\nvalue-mismatch grouping (first 4 name components):")
        for g, n in sorted(groups.items(), key=lambda x: -x[1]):
            print(f"  {n:5d}  {g}")

    bad = only_a or only_b or shape_diff or dtype_diff or sha_diff
    if not bad:
        print("\nIDENTICAL: every tensor matches by name, shape, dtype and value.")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
