"""Dump full-precision scalars from a tensorboard event file.

Usage: python read_tb.py <event_file_or_dir> [tag_substr]

Prints "tag<TAB>step<TAB>value(repr)" so a reference run whose stdout only logs
4 decimals (`Loss: 8.7231 (Video: 1.1622, Action: 7.5609)`) can be compared at
full float32 precision. This is not optional for parity work: 4 decimals cannot
distinguish a 3e-5 run-to-run band from a 2e-4 real divergence.

Typical use (BASE side):
    python read_tb.py /tmp/base_ref_r1/checkpoints/<exp>/<run>/tensorboard_logs video
    python read_tb.py <run_dir>            # every scalar tag

If the reference repo reports to wandb by default, override it to tensorboard
for the parity run (`--report_to tensorboard`) — local event files are far
easier to read back at full precision than the wandb API.

No EDIT-ME markers: this file is project-independent.
"""
import glob
import os
import sys

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

path = sys.argv[1]
want = sys.argv[2] if len(sys.argv) > 2 else ""
if os.path.isdir(path):
    cands = glob.glob(os.path.join(path, "**", "events.out.tfevents*"), recursive=True)
    if not cands:
        sys.exit(f"no events.out.tfevents* under {path}")
    path = sorted(cands)[-1]
    print(f"# event file: {path}", file=sys.stderr)
ea = EventAccumulator(path, size_guidance={"scalars": 0})
ea.Reload()
for tag in sorted(ea.Tags()["scalars"]):
    if want and want not in tag:
        continue
    for ev in ea.Scalars(tag):
        print(f"{tag}\t{ev.step}\t{ev.value!r}")
