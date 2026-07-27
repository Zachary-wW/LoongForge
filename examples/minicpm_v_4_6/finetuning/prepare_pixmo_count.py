#!/usr/bin/env python3
# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Prepare a shared PixMo-Count subset for MiniCPM-V-4.6 SFT comparisons."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
from io import BytesIO
from pathlib import Path
from typing import Any
from urllib.request import urlopen

from datasets import load_dataset
from PIL import Image


QUESTION_TEMPLATE = "Count the {label} in the image."


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _normalize_message_content(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    for row in rows:
        for message in row.get("messages", []):
            if isinstance(message.get("content"), str):
                message["content"] = [
                    {"type": "text", "text": message["content"]},
                ]
    return rows


def _swift_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    swift_rows = []
    for row in rows:
        swift_row = dict(row)
        swift_row.pop("images", None)
        swift_rows.append(swift_row)
    return swift_rows


def _is_valid_image(data: bytes) -> bool:
    try:
        with Image.open(BytesIO(data)) as image:
            image.verify()
    except Exception:
        return False
    return True


def _download(
    url: str,
    path: Path,
    expected_sha256: str | None,
    timeout: int,
    verify_sha256: bool,
) -> bool:
    if path.is_file():
        if verify_sha256 and expected_sha256:
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if digest == expected_sha256:
                return True
        elif _is_valid_image(path.read_bytes()):
            return True

    try:
        with urlopen(url, timeout=timeout) as response:
            data = response.read()
    except Exception as exc:
        print(f"skip download failed: {url} ({exc})", flush=True)
        return False

    if not _is_valid_image(data):
        print(f"skip invalid image: {url}", flush=True)
        return False

    if verify_sha256 and expected_sha256:
        digest = hashlib.sha256(data).hexdigest()
        if digest != expected_sha256:
            print(f"skip sha256 mismatch: {url}", flush=True)
            return False

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return True


def _format_points(points: dict[str, Any]) -> str:
    xs = points.get("x", []) if points else []
    ys = points.get("y", []) if points else []
    point_tags = [
        f"<point>{round(float(x), 3)} {round(float(y), 3)}</point>"
        for x, y in zip(xs, ys)
    ]
    return "".join(point_tags)


def _build_row(image_path: Path, label: str, count: int, points: dict[str, Any]) -> dict[str, Any]:
    question = QUESTION_TEMPLATE.format(label=label)
    answer = f"{_format_points(points)}{count}"
    return {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": str(image_path)},
                    {"type": "text", "text": question},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": answer}],
            },
        ],
        "images": [str(image_path)],
    }


def _prepare_one(
    item: dict[str, Any],
    index: int,
    image_dir: Path,
    timeout: int,
    verify_sha256: bool,
) -> dict[str, Any] | None:
    image_url = item["image_url"]
    image_sha256 = item.get("image_sha256")
    suffix = os.path.splitext(image_url.split("?", 1)[0])[1] or ".jpg"
    image_path = image_dir / f"{index:06d}_{image_sha256 or 'image'}{suffix}"
    if not _download(
        image_url,
        image_path,
        image_sha256,
        timeout,
        verify_sha256,
    ):
        return None

    row = _build_row(
        image_path=image_path,
        label=item["label"],
        count=item["count"],
        points=item.get("points") or {},
    )
    row["metadata"] = {
        "source_dataset": "allenai/pixmo-count",
        "source_index": index,
        "image_url": image_url,
        "image_sha256": image_sha256,
    }
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="allenai/pixmo-count")
    parser.add_argument("--split", default="train")
    parser.add_argument("--output-root", default="/data/weizhihao/agent/datasets/pixmo-count")
    parser.add_argument("--num-samples", type=int, default=128)
    parser.add_argument("--scan-limit", type=int, default=5000)
    parser.add_argument("--timeout", type=int, default=12)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--verify-sha256", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    output_root = Path(args.output_root)
    image_dir = output_root / "images"
    loongforge_path = output_root / "loongforge" / "pixmo_count_openai.jsonl"
    swift_path = output_root / "swift" / "pixmo_count_openai.jsonl"

    existing = _read_jsonl(loongforge_path)
    if not args.force and len(existing) >= args.num_samples and swift_path.is_file():
        print(f"dataset already prepared: {loongforge_path} ({len(existing)} rows)")
        existing = _normalize_message_content(existing[: args.num_samples])
        _write_jsonl(loongforge_path, existing)
        swift_rows = _swift_rows(existing)
        _write_jsonl(swift_path, swift_rows)
        return

    ds = load_dataset(args.dataset, split=args.split)
    rows: list[dict[str, Any]] = []
    limit = min(args.scan_limit, len(ds))

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [
            executor.submit(
                _prepare_one,
                dict(ds[index]),
                index,
                image_dir,
                args.timeout,
                args.verify_sha256,
            )
            for index in range(limit)
        ]
        for seen, future in enumerate(concurrent.futures.as_completed(futures), start=1):
            row = future.result()
            if row is not None:
                row["metadata"]["source_dataset"] = args.dataset
                rows.append(row)
                if len(rows) % 16 == 0:
                    print(f"prepared {len(rows)} rows after {seen} attempts", flush=True)
            if len(rows) >= args.num_samples:
                break

        for future in futures:
            future.cancel()

    rows.sort(key=lambda row: row["metadata"]["source_index"])

    if len(rows) < args.num_samples:
        raise RuntimeError(
            f"Only prepared {len(rows)} rows from {limit} scanned samples; "
            "increase --scan-limit or check image download connectivity."
        )

    rows = _normalize_message_content(rows)
    _write_jsonl(loongforge_path, rows)
    _write_jsonl(swift_path, _swift_rows(rows))
    print(f"wrote {len(rows)} rows")
    print(f"loongforge: {loongforge_path}")
    print(f"swift: {swift_path}")


if __name__ == "__main__":
    main()
