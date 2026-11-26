import ast
import importlib
import shutil
import subprocess
import sys
from pathlib import Path

from validate.base import validator_source_sha256
from validate.intake_check import IntakeChecker

ROOT = Path(__file__).parents[2]


def test_canonical_package_layout_and_no_duplicate_legacy_shells() -> None:
    for name in ("engine", "stages", "rsi", "profiles"):
        assert not (ROOT / name).exists()
    assert (ROOT / "core/__init__.py").read_bytes() == b""
    assert {p.name for p in (ROOT / "core/stages").iterdir() if p.is_dir() and p.name != "__pycache__"} == {
        "f1_intake", "f2_anatomy", "f3_scaffold", "f4_adapt", "f5_verify", "f6_deliver",
    }
    assert not list((ROOT / "core/stages/f5_verify").glob("*_check.py"))


def test_orchestrator_imports_without_pydantic() -> None:
    # Test actual transitive imports, not just a grep of direct dependencies.
    result = subprocess.run([sys.executable, "-c", """
import sys
class DenyPydantic:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'pydantic' or fullname.startswith('pydantic.'):
            raise ImportError('pydantic forbidden in orchestrator')
sys.meta_path.insert(0, DenyPydantic())
from core.orchestrator import graph, scheduler, judger, invalidation, isolation
assert not any(k.startswith('pydantic') for k in sys.modules)
"""], cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    for path in (ROOT / "core/orchestrator").glob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.ImportFrom):
                assert not (node.module or "").startswith("core.contracts")


def test_validator_hash_binds_relative_source_paths_not_checkout_root(tmp_path: Path) -> None:
    module = tmp_path / "helper.py"
    module.write_text("VALUE = 1\n")
    before = validator_source_sha256(IntakeChecker, extra_sources=[module])
    module.write_text("VALUE = 2\n")
    assert before != validator_source_sha256(IntakeChecker, extra_sources=[module])
    for name in ("intake", "anatomy", "construct", "parity", "train_resume", "eval"):
        cls = importlib.import_module(f"validate.{name}_check")
        assert cls is not None


def test_composite_hash_is_stable_when_checkout_moves_and_binds_v2_helpers(tmp_path: Path) -> None:
    from validate.intake_check import _intake_dependency_sources
    sources = {p.relative_to(ROOT).as_posix() for p in _intake_dependency_sources()}
    assert {"core/contracts/intake_v2.py", "core/stages/f1_intake/handoff.py"} <= sources
    for directory in ("core", "validate"):
        shutil.copytree(ROOT / directory, tmp_path / directory, ignore=shutil.ignore_patterns("__pycache__"))
    result = subprocess.run(
        [sys.executable, "-c", "from validate.intake_check import IntakeChecker; print(IntakeChecker().validator_sha256())"],
        cwd=tmp_path, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == IntakeChecker().validator_sha256()
