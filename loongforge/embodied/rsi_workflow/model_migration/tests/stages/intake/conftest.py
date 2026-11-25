from __future__ import annotations

import sys
from pathlib import Path

# Allow sibling `helpers.py` imports without turning `tests/stages` into a
# top-level `stages` package that shadows the real `stages/` package.
_THIS_DIR = str(Path(__file__).resolve().parent)
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)
