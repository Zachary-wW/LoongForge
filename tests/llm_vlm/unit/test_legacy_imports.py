from pathlib import Path


def test_loongforge_has_no_unregistered_legacy_imports():
    package_root = Path(__file__).resolve().parents[3] / "loongforge"
    allowed = {
        "train/checkpointing.py",
    }
    offenders = []
    for path in package_root.rglob("*.py"):
        relative = path.relative_to(package_root).as_posix()
        has_import = any(
            line.lstrip().startswith(("from megatron.legacy", "import megatron.legacy"))
            for line in path.read_text(encoding="utf-8").splitlines()
        )
        if has_import and relative not in allowed:
            offenders.append(relative)
    assert offenders == []
