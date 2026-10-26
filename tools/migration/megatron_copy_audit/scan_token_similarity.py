#!/usr/bin/env python3
# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
"""Token-level Megatron provenance scan for heavily modified LoongForge files."""

from __future__ import annotations

import argparse
import ast
import csv
import io
import json
import re
import tokenize
from collections import Counter, defaultdict
from pathlib import Path

VERSIONS = [
    "core_v0.10.0",
    "core_v0.12.0",
    "core_v0.13.0",
    "core_v0.14.0",
    "core_v0.15.0",
    "core_v0.19.2",
]
SHINGLE = 12
MIN_SHARED = 8
MAX_DF = 40


def tokens(path: Path) -> list[str]:
    try:
        text = ast.unparse(ast.parse(path.read_text(encoding="utf-8")))
    except (UnicodeDecodeError, SyntaxError, ValueError):
        return []
    output = []
    for tok in tokenize.generate_tokens(io.StringIO(text).readline):
        if tok.type in {
            tokenize.NAME,
            tokenize.OP,
            tokenize.NUMBER,
            tokenize.STRING,
        }:
            value = re.sub(r"\s+", " ", tok.string.strip())
            if value:
                output.append(value)
    return output


def shingles(items: list[str]) -> set[str]:
    if len(items) < SHINGLE:
        return set()
    return {"\0".join(items[index : index + SHINGLE]) for index in range(len(items) - SHINGLE + 1)}


def files(root: Path):
    yield from sorted(root.rglob("*.py"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--loongforge-root", type=Path, required=True)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--versions", default=",".join(VERSIONS))
    args = parser.parse_args()
    target_root = args.loongforge_root.resolve()
    reference_root = args.reference_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    versions = [version.strip() for version in args.versions.split(",") if version.strip()]

    target = {}
    for path in files(target_root / "loongforge"):
        rel = path.relative_to(target_root).as_posix()
        item_tokens = tokens(path)
        target[rel] = {
            "tokens": len(item_tokens),
            "shingles": shingles(item_tokens),
            "matches": [],
        }

    for version in versions:
        root = reference_root / version
        ref_paths = list(files(root))
        refs = {}
        inverted = defaultdict(list)
        for idx, path in enumerate(ref_paths):
            rel = path.relative_to(root).as_posix()
            item_tokens = tokens(path)
            refs[idx] = {
                "path": rel,
                "tokens": len(item_tokens),
                "shingles": shingles(item_tokens),
            }
            for item in refs[idx]["shingles"]:
                inverted[item].append(idx)
        inverted = {key: indexes for key, indexes in inverted.items() if len(indexes) <= MAX_DF}

        for rel, target_item in target.items():
            counts: Counter[int] = Counter()
            for item in target_item["shingles"]:
                for ref_idx in inverted.get(item, ()):
                    counts[ref_idx] += 1
            for ref_idx, shared in counts.most_common(10):
                if shared < MIN_SHARED:
                    continue
                ref = refs[ref_idx]
                t_size = max(1, len(target_item["shingles"]))
                r_size = max(1, len(ref["shingles"]))
                target_item["matches"].append(
                    {
                        "version": version,
                        "source": ref["path"],
                        "shared": shared,
                        "target_tokens": target_item["tokens"],
                        "source_tokens": ref["tokens"],
                        "target_coverage": round(shared / t_size, 4),
                        "source_coverage": round(shared / r_size, 4),
                        "containment": round(shared / min(t_size, r_size), 4),
                    }
                )

    records = []
    for rel, target_item in target.items():
        matches = sorted(
            target_item["matches"],
            key=lambda item: (
                item["containment"],
                item["source_coverage"],
                item["shared"],
            ),
            reverse=True,
        )
        best = matches[0] if matches else {}
        records.append(
            {
                "loongforge_path": rel,
                "best_version": best.get("version", ""),
                "best_source": best.get("source", ""),
                "best_containment": best.get("containment", 0),
                "best_source_coverage": best.get("source_coverage", 0),
                "best_target_coverage": best.get("target_coverage", 0),
                "best_shared": best.get("shared", 0),
                "matches": matches[:12],
            }
        )
    records.sort(
        key=lambda item: (
            item["best_containment"],
            item["best_source_coverage"],
            item["best_target_coverage"],
        ),
        reverse=True,
    )
    out = output_dir / "token_similarity_results.json"
    out.write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n")
    fields = [
        "loongforge_path",
        "best_version",
        "best_source",
        "best_containment",
        "best_source_coverage",
        "best_target_coverage",
        "best_shared",
    ]
    with (output_dir / "token_similarity_results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows({key: row.get(key, "") for key in fields} for row in records)
    print(f"targets={len(records)}")
    for record in records[:100]:
        if record["best_containment"] < 0.03:
            break
        print(
            f"{record['best_containment']:.3f}\t"
            f"{record['best_source_coverage']:.3f}\t"
            f"{record['best_version']}\t"
            f"{record['loongforge_path']}\t{record['best_source']}"
        )


if __name__ == "__main__":
    main()
