#!/bin/bash
# Reap a finished-but-still-holding run before launching the next parity A/B.
# Project-independent. Run this between EVERY pair of runs.
#
#   ./reap_gpus.sh "loongforge/embodied/train.py"      # match by command line
#   ./reap_gpus.sh "train/train.py" 120                # custom wait budget (s)
#
# Why: DeepSpeed + accelerate runs routinely print the last step, save, and then
# hang in teardown (`dist` already torn down, GC re-enters print_rank_0 ->
# "AssertionError: DeepSpeed backend not set"). The processes keep every GPU's
# memory. The next run then fails with a nonsense OOM ("tried to allocate 18 MiB")
# while a zombie holds 70 GB — and you will spend an hour blaming your new code.
#
# Also: `nvidia-smi` reports HOST pids, which do not exist inside a container
# (`ps -p` exits 1, /proc/<pid> is absent). Always match by command line instead.
set -u
PAT="${1:?usage: $0 <pgrep-pattern> [wait_seconds]}"
WAIT="${2:-60}"

# Never match this script, its parent shell, or the pgrep pipeline itself: their
# argv all contain $PAT, so a naive `pkill -f "$PAT"` kills the reaper and reports
# success while the real run keeps holding the GPUs.
self=" $$ $PPID "
victims=""
for p in $(pgrep -f -- "$PAT" 2>/dev/null); do
    case "$self" in *" $p "*) continue ;; esac
    [ -r "/proc/$p/cmdline" ] || continue           # already exited
    cmd=$(tr '\0' ' ' < "/proc/$p/cmdline" 2>/dev/null)
    [ -n "$cmd" ] || continue
    case "$cmd" in *reap_gpus.sh*|pgrep*) continue ;; esac
    victims="$victims $p"
    echo "  CANDIDATE $p ${cmd:0:140}"
done

echo "### processes matching: $PAT"
if [ -z "$victims" ]; then
    echo "  (none)"
else
    kill -9 $victims 2>/dev/null
    echo "### sent SIGKILL to:$victims"
fi

for i in $(seq 1 "$WAIT"); do
    sleep 1
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | paste -sd+ | bc)
    if [ "$used" -lt 100 ]; then
        echo "### GPUs clear after ${i}s (total used=${used} MiB)"
        nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
        exit 0
    fi
done

echo "### STILL HELD after ${WAIT}s — do NOT launch the next run yet:"
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
echo "### remaining matches (widen the pattern, or another user holds the GPUs):"
pgrep -fa -- "$PAT" | grep -v -- "reap_gpus.sh" || echo "  (none)"
exit 1
