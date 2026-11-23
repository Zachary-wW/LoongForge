import subprocess
import sys


def test_mcore_llm_entries_do_not_import_legacy_runtime():
    code = """
import importlib
import sys

importlib.import_module("loongforge.train.pretrain.pretrain_llm")
importlib.import_module("loongforge.train.sft.sft_llm")
for name in ("loongforge.train.training_utils", "loongforge.train.checkpointing"):
    assert name not in sys.modules, name
"""
    subprocess.run([sys.executable, "-c", code], check=True)
