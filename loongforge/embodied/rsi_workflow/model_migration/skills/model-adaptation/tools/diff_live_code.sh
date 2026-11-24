#!/bin/bash
# Live-code diff between a BASE file and its TRANSPLANT counterpart (Rule 2/3).
# Normalizes both with codenorm.py first, so reflow / docstrings / comments do not
# show up and only real code changes remain.
#
# Usage: ./diff_live_code.sh BASE/models/foo.py TRANS/model/foo.py
#
# Then classify EVERY remaining hunk as one of:
#   dead-code  — a branch that never executes on the runtime path (prove it)
#   probe      — parity/NVTX instrumentation gated off by default
#   live-op    — a real math difference. Only these matter.
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[ $# -eq 2 ] || { echo "usage: $0 <base.py> <trans.py>" >&2; exit 2; }

A=$(mktemp /tmp/codenorm_base_XXXX.txt)
B=$(mktemp /tmp/codenorm_trans_XXXX.txt)
trap 'rm -f "$A" "$B"' EXIT

python "$HERE/codenorm.py" "$1" > "$A" || exit 1
python "$HERE/codenorm.py" "$2" > "$B" || exit 1

echo "### normalized line counts: base=$(wc -l < "$A")  trans=$(wc -l < "$B")"
diff -u "$A" "$B" && echo "### IDENTICAL after normalization (all diff was formatting)"
