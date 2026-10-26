#!/usr/bin/env python3
# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
"""Locate LoongForge files that retain code copied from older Megatron-LM."""

from __future__ import annotations

import ast
import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

VERSIONS = ["core_v0.10.0", "core_v0.12.0", "core_v0.13.0", "core_v0.14.0", "core_v0.15.0", "core_v0.19.2"]
SHINGLE_SIZE = 6
MIN_SHARED = 18
MAX_SHINGLE_DOC_FREQUENCY = 40


def normalized_lines(path: Path) -> list[str]:
    """Return AST-unparsed, comment-free lines for stable structural matching."""
    try:
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        text = ast.unparse(tree)
    except (UnicodeDecodeError, SyntaxError, ValueError):
        return []
    lines = []
    for raw in text.splitlines():
        line = re.sub(r"\s+", " ", raw.strip())
        if not line or line in {"pass", "..."}:
            continue
        lines.append(line)
    return lines


def shingles(lines: list[str]) -> set[str]:
    if len(lines) < SHINGLE_SIZE:
        return set()
    return {"\n".join(lines[index : index + SHINGLE_SIZE]) for index in range(len(lines) - SHINGLE_SIZE + 1)}


def iter_python(root: Path):
    yield from sorted(root.rglob("*.py"))


def aliases_for(rel_path: str) -> list[str]:
    """Known historical path shifts used as extra candidate seeds."""
    aliases = [rel_path]
    parts = Path(rel_path).parts
    if parts[:1] == ("loongforge",):
        tail = Path(*parts[1:])
    else:
        tail = Path(rel_path)
    candidates = {
        str(tail),
        str(Path("megatron") / tail),
        str(Path("megatron/core") / tail),
        str(Path("megatron/training") / tail),
    }
    name = Path(rel_path).name
    if name in {"pretrain_llm.py", "pretrain_vlm.py", "sft_llm.py"}:
        candidates.add("pretrain_gpt.py")
    if name == "base_gpt_model.py":
        candidates.add("megatron/core/models/gpt/gpt_model.py")
    if name in {"language_transformer_block.py", "vision_transformer_block.py"}:
        candidates.add("megatron/core/transformer/transformer_block.py")
    if name == "multi_latent_attention.py":
        candidates.add("megatron/core/transformer/multi_latent_attention.py")
    if name in {"local_attention.py", "attention.py"}:
        candidates.add("megatron/core/transformer/attention.py")
    if name in {"local_norm.py"}:
        candidates.add("megatron/core/transformer/norm.py")
    return sorted(candidates | {alias for alias in candidates if Path(alias).name == name})


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

    targets = list(iter_python(target_root / "loongforge"))
    target_data = {}
    for path in targets:
        rel = path.relative_to(target_root).as_posix()
        lines = normalized_lines(path)
        target_data[rel] = {
            "path": path,
            "lines": len(lines),
            "shingles": shingles(lines),
            "matches": [],
        }

    for version in versions:
        root = reference_root / version
        ref_paths = list(iter_python(root))
        ref_meta = {}
        inverted = defaultdict(list)
        for index, path in enumerate(ref_paths):
            rel = path.relative_to(root).as_posix()
            lines = normalized_lines(path)
            ref_meta[index] = {
                "path": rel,
                "lines": len(lines),
                "shingles": shingles(lines),
            }
            for shingle in ref_meta[index]["shingles"]:
                inverted[shingle].append(index)

        # Remove common boilerplate shingles; they only inflate candidate counts.
        inverted = {key: indexes for key, indexes in inverted.items() if len(indexes) <= MAX_SHINGLE_DOC_FREQUENCY}
        for target in target_data.values():
            counts: Counter[int] = Counter()
            for shingle in target["shingles"]:
                for ref_index in inverted.get(shingle, ()):
                    counts[ref_index] += 1

            seeded = set()
            basename = target["path"].name
            aliases = set(aliases_for(target["path"].relative_to(target_root).as_posix()))
            for index, ref in ref_meta.items():
                if ref["path"] in aliases or ref["path"].endswith("/" + basename):
                    seeded.add(index)
            for index in seeded:
                counts[index] += 1

            ranked = sorted(counts.items(), key=lambda item: item[1], reverse=True)[:8]
            for ref_index, shared in ranked:
                ref = ref_meta[ref_index]
                target_size = max(1, len(target["shingles"]))
                ref_size = max(1, len(ref["shingles"]))
                if shared < MIN_SHARED and ref["path"] not in aliases:
                    continue
                target_cov = min(1.0, shared / target_size)
                ref_cov = min(1.0, shared / ref_size)
                containment = shared / max(1, min(target_size, ref_size))
                jaccard = shared / max(1, target_size + ref_size - shared)
                target["matches"].append(
                    {
                        "version": version,
                        "source": ref["path"],
                        "shared": shared,
                        "target_lines": target["lines"],
                        "source_lines": ref["lines"],
                        "target_coverage": round(target_cov, 4),
                        "source_coverage": round(ref_cov, 4),
                        "containment": round(containment, 4),
                        "jaccard": round(jaccard, 4),
                        "path_seed": ref["path"] in aliases,
                    }
                )

    records = []
    for rel, target in target_data.items():
        matches = sorted(
            target["matches"],
            key=lambda item: (item["containment"], item["source_coverage"], item["shared"]),
            reverse=True,
        )
        best = matches[0] if matches else {}
        old_matches = [item for item in matches if item["version"] != "core_v0.19.2"]
        best_old = max(
            old_matches,
            key=lambda item: (item["containment"], item["source_coverage"], item["shared"]),
            default={},
        )
        records.append(
            {
                "loongforge_path": rel,
                "best_version": best.get("version", ""),
                "best_source": best.get("source", ""),
                "best_containment": best.get("containment", 0),
                "best_source_coverage": best.get("source_coverage", 0),
                "best_target_coverage": best.get("target_coverage", 0),
                "best_shared": best.get("shared", 0),
                "best_old_version": best_old.get("version", ""),
                "best_old_source": best_old.get("source", ""),
                "best_old_containment": best_old.get("containment", 0),
                "best_old_source_coverage": best_old.get("source_coverage", 0),
                "best_old_target_coverage": best_old.get("target_coverage", 0),
                "best_old_shared": best_old.get("shared", 0),
                "matches": matches[:12],
            }
        )

    records.sort(
        key=lambda item: (
            item["best_old_containment"],
            item["best_old_source_coverage"],
            item["best_old_target_coverage"],
        ),
        reverse=True,
    )
    (output_dir / "similarity_results.json").write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n")
    with (output_dir / "similarity_results.csv").open("w", newline="") as handle:
        fieldnames = [
            "loongforge_path",
            "best_old_version",
            "best_old_source",
            "best_old_containment",
            "best_old_source_coverage",
            "best_old_target_coverage",
            "best_old_shared",
            "best_version",
            "best_source",
            "best_containment",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow({key: record.get(key, "") for key in fieldnames})

    print(f"targets={len(records)}")
    for record in records[:80]:
        if record["best_old_containment"] < 0.08:
            break
        print(
            f"{record['best_old_containment']:.3f}\t"
            f"{record['best_old_source_coverage']:.3f}\t"
            f"{record['best_old_version']}\t"
            f"{record['loongforge_path']}\t{record['best_old_source']}"
        )


if __name__ == "__main__":
    main()
