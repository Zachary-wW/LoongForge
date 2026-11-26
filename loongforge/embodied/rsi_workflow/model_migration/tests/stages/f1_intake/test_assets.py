from __future__ import annotations

from pathlib import Path

from core.contracts.common import AssetSpec
from core.stages.f1_intake.assets import triage_assets

from helpers import make_asset_file


def test_triage_assets_ready_waiting_and_blocked_paths(tmp_path: Path) -> None:
    ckpt_path, ckpt_sha = make_asset_file(tmp_path, "assets/model.safetensors", b"weights-v1")
    stat_path, stat_sha = make_asset_file(tmp_path, "assets/norm_stats.json", b'{"mean":0}\n')

    # 1) All present and matching digests -> ready (and untouched/read-only)
    ready_specs = [
        AssetSpec(
            id="checkpoint",
            kind="checkpoint",
            required_for=["f1_startup_probe"],
            source="hf://example/pi05",
            revision="main",
            expected_sha256=ckpt_sha,
            access_policy="owner_provided",
            local_path=str(ckpt_path),
        ),
        AssetSpec(
            id="statistics",
            kind="statistics",
            required_for=["f1_startup_probe"],
            source="local://stats",
            revision="v1",
            expected_sha256=stat_sha,
            access_policy="owner_provided",
            local_path=str(stat_path),
        ),
    ]
    report = triage_assets(ready_specs, required_scope="f1_startup_probe")
    assert report.overall_status == "ready"
    assert report.blocked_reasons == ()
    assert report.waiting_reasons == ()
    assert all(rec.observed_sha256 == rec.expected_sha256 for rec in report.records)

    # 2) Declared identity (source + expected_sha256) but missing local_path -> waiting / WAITING_ASSET
    waiting_specs = [
        AssetSpec(
            id="dataset",
            kind="dataset",
            required_for=["f1_startup_probe"],
            source="hf://example/libero",
            revision="main",
            expected_sha256=ckpt_sha,
            access_policy="owner_provided",
            local_path=None,
        )
    ]
    waiting_report = triage_assets(waiting_specs, required_scope="f1_startup_probe")
    assert waiting_report.overall_status == "waiting"
    assert "WAITING_ASSET" in waiting_report.waiting_reasons
    assert waiting_report.records[0].status == "waiting"

    # 3) Local file exists but SHA256 mismatches -> blocked
    bad_sha = "1" * 64
    mismatch_specs = [
        AssetSpec(
            id="checkpoint",
            kind="checkpoint",
            required_for=["f1_startup_probe"],
            source="hf://example/pi05",
            revision="main",
            expected_sha256=bad_sha,
            access_policy="owner_provided",
            local_path=str(ckpt_path),
        )
    ]
    mismatch_report = triage_assets(mismatch_specs, required_scope="f1_startup_probe")
    assert mismatch_report.overall_status == "blocked"
    assert "ASSET_DIGEST_MISMATCH" in mismatch_report.blocked_reasons

    # 4) Unknown required asset (missing both local_path and declared source/expected_sha256) -> blocked
    unknown_specs = [
        AssetSpec(
            id="simulator",
            kind="simulator",
            required_for=["f1_startup_probe"],
            source=None,
            revision=None,
            expected_sha256=None,
            access_policy="owner_provided",
            local_path=None,
        )
    ]
    unknown_report = triage_assets(unknown_specs, required_scope="f1_startup_probe")
    assert unknown_report.overall_status == "blocked"
    assert "UNKNOWN_REQUIRED_ASSET" in unknown_report.blocked_reasons


def test_triage_assets_rejects_top_level_nested_and_dangling_symlinks(
    tmp_path: Path,
) -> None:
    real_file, real_sha = make_asset_file(tmp_path, "real/weights.bin", b"real-weights")

    # 1) Top-level symlink pointing to a real file must be blocked BEFORE resolve()
    top_link = tmp_path / "links" / "top_weights.bin"
    top_link.parent.mkdir(parents=True, exist_ok=True)
    top_link.symlink_to(real_file)
    top_report = triage_assets(
        [
            AssetSpec(
                id="checkpoint",
                kind="checkpoint",
                required_for=["f1_startup_probe"],
                source="hf://example/pi05",
                revision="main",
                expected_sha256=real_sha,
                access_policy="owner_provided",
                local_path=str(top_link),
            )
        ],
        required_scope="f1_startup_probe",
    )
    assert top_report.overall_status == "blocked"
    assert "ASSET_SYMLINK_REJECTED" in top_report.blocked_reasons
    assert top_report.records[0].status == "blocked"

    # 2) Dangling symlink must be blocked (not treated as missing/waiting!)
    dangling_link = tmp_path / "links" / "dangling.bin"
    dangling_link.symlink_to(tmp_path / "does_not_exist.bin")
    dangling_report = triage_assets(
        [
            AssetSpec(
                id="dataset",
                kind="dataset",
                required_for=["f1_startup_probe"],
                source="hf://example/libero",
                revision="main",
                expected_sha256=real_sha,
                access_policy="owner_provided",
                local_path=str(dangling_link),
            )
        ],
        required_scope="f1_startup_probe",
    )
    assert dangling_report.overall_status == "blocked"
    assert "ASSET_SYMLINK_REJECTED" in dangling_report.blocked_reasons

    # 3) Nested symlink inside a directory asset must be blocked
    asset_dir = tmp_path / "dataset_dir"
    (asset_dir / "nested").mkdir(parents=True)
    (asset_dir / "meta.json").write_text('{"ok": true}\n', encoding="utf-8")
    (asset_dir / "nested" / "escaped.bin").symlink_to(real_file)
    nested_report = triage_assets(
        [
            AssetSpec(
                id="dataset_nested",
                kind="dataset",
                required_for=["f1_startup_probe"],
                source="hf://example/libero",
                revision="main",
                expected_sha256=real_sha,
                access_policy="owner_provided",
                local_path=str(asset_dir),
            )
        ],
        required_scope="f1_startup_probe",
    )
    assert nested_report.overall_status == "blocked"
    assert "ASSET_SYMLINK_REJECTED" in nested_report.blocked_reasons
