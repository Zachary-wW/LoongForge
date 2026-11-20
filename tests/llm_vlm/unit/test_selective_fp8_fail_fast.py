from types import SimpleNamespace

from loongforge.train.parser import _has_selective_fp8_config


def test_selective_fp8_detects_nested_foundation():
    config = SimpleNamespace(
        selective_fp8=False,
        foundation=SimpleNamespace(selective_fp8=True),
    )
    assert _has_selective_fp8_config({"model": config})


def test_selective_fp8_absent_remains_disabled():
    config = SimpleNamespace(selective_fp8=False, foundation=SimpleNamespace(selective_fp8=False))
    assert not _has_selective_fp8_config({"model": config})
