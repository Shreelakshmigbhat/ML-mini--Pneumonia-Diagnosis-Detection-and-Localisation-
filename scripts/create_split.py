"""Create one reproducible, stratified train/validation/test assignment."""

from __future__ import annotations

import argparse
import csv
import hashlib
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


SEED = 42
SPLIT_RATIOS = (("train", 70), ("validation", 20), ("test", 10))
INPUT_PATH = Path("metadata") / "pneumonia_metadata.csv"
OUTPUT_PATH = Path("metadata") / "split_metadata.csv"
STATS_PATH = Path("metadata") / "SPLIT_STATS.md"
OUTPUT_COLUMNS = (
    "image_id",
    "image_path",
    "label",
    "split",
    "x",
    "y",
    "width",
    "height",
)
REQUIRED_INPUT_COLUMNS = (
    "image_id",
    "image_path",
    "original_label",
    "binary_label",
    "x",
    "y",
    "width",
    "height",
)


def _split_counts(class_count: int) -> dict[str, int]:
    allocations = {
        split: class_count * percentage // 100
        for split, percentage in SPLIT_RATIOS
    }
    remainders = {
        split: class_count * percentage % 100
        for split, percentage in SPLIT_RATIOS
    }
    unallocated = class_count - sum(allocations.values())
    priority = {split: index for index, (split, _) in enumerate(SPLIT_RATIOS)}
    for split in sorted(
        remainders, key=lambda name: (-remainders[name], priority[name])
    )[:unallocated]:
        allocations[split] += 1
    return allocations


def _stable_order_key(label: str, image_id: str) -> bytes:
    return hashlib.sha256(f"{SEED}\0{label}\0{image_id}".encode("utf-8")).digest()


def _read_input(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as csv_file:
        reader = csv.DictReader(csv_file)
        if reader.fieldnames is None:
            raise ValueError(f"Metadata CSV has no header: {path}")
        missing = set(REQUIRED_INPUT_COLUMNS) - set(reader.fieldnames)
        if missing:
            raise ValueError(f"Metadata CSV is missing columns: {sorted(missing)}")
        rows = list(reader)

    seen_ids: set[str] = set()
    for row in rows:
        image_id = row["image_id"].strip()
        if not image_id:
            raise ValueError("Metadata contains an empty image_id")
        key = image_id.casefold()
        if key in seen_ids:
            raise ValueError(f"Duplicate image_id in metadata: {image_id}")
        seen_ids.add(key)
        if row["binary_label"] not in {"0", "1"}:
            raise ValueError(
                f"Unexpected binary_label={row['binary_label']!r} "
                f"for image {image_id}"
            )
    if not rows:
        raise ValueError(f"Metadata CSV contains no rows: {path}")
    return rows


def _make_assignments(
    rows: list[dict[str, str]],
) -> dict[str, str]:
    by_label: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_label[row["binary_label"]].append(row)

    assignments: dict[str, str] = {}
    for label, class_rows in sorted(by_label.items()):
        ordered_rows = sorted(
            class_rows,
            key=lambda row: (
                _stable_order_key(label, row["image_id"]),
                row["image_id"],
            ),
        )
        offsets = 0
        for split, _ in SPLIT_RATIOS:
            split_count = _split_counts(len(class_rows))[split]
            for row in ordered_rows[offsets : offsets + split_count]:
                assignments[row["image_id"]] = split
            offsets += split_count
        if offsets != len(class_rows):
            raise AssertionError(f"Split allocation did not cover label {label}")
    if len(assignments) != len(rows):
        raise AssertionError("Not every metadata image received one split")
    return assignments


def _validate_no_overlap(rows: list[dict[str, str]], assignments: dict[str, str]) -> None:
    ids_by_split: dict[str, set[str]] = {split: set() for split, _ in SPLIT_RATIOS}
    for row in rows:
        ids_by_split[assignments[row["image_id"]]].add(row["image_id"])
    for index, (left, _) in enumerate(SPLIT_RATIOS):
        for right, _ in SPLIT_RATIOS[index + 1 :]:
            overlap = ids_by_split[left] & ids_by_split[right]
            if overlap:
                raise ValueError(
                    f"Image ID overlap between {left} and {right}: "
                    f"{sorted(overlap)[:5]}"
                )


def _write_stats(path: Path, rows: list[dict[str, str]], assignments: dict[str, str]) -> None:
    counts: dict[str, Counter[str]] = {
        split: Counter() for split, _ in SPLIT_RATIOS
    }
    class_totals: Counter[str] = Counter()
    for row in rows:
        split = assignments[row["image_id"]]
        label = row["binary_label"]
        counts[split][label] += 1
        class_totals[label] += 1

    total = len(rows)
    label_names = {"1": "Pneumonia", "0": "Healthy"}
    lines = [
        "# Fixed train/validation/test split statistics",
        "",
        f"Generated from `{INPUT_PATH.as_posix()}` by "
        "`scripts/create_split.py`. Seed: **42**. Splits are stratified by "
        "`binary_label` at the image/patient-ID level.",
        "",
        "The metadata contains no separate patient-ID field: `image_id` is the "
        "RSNA `patientId` carried forward by Stage 1. Each ID occurs exactly "
        "once in the source metadata and is assigned to exactly one split. "
        "The script rejects duplicate IDs and verifies pairwise split disjointness.",
        "",
        "Per-class counts are allocated as closely as possible to 70% / 20% / "
        "10% using largest remainders. Remainder ties use train, validation, "
        "test priority. A SHA-256 ordering of seed, class, and image ID makes "
        "the assignment deterministic without depending on row order or a "
        "third-party random-number implementation.",
        "",
        "| Split | Images | % of dataset | Pneumonia | Healthy | Pneumonia within split | Healthy within split |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for split, _ in SPLIT_RATIOS:
        split_total = sum(counts[split].values())
        positive = counts[split]["1"]
        negative = counts[split]["0"]
        lines.append(
            f"| {split} | {split_total:,} | {100 * split_total / total:.2f}% | "
            f"{positive:,} | {negative:,} | "
            f"{100 * positive / split_total:.2f}% | "
            f"{100 * negative / split_total:.2f}% |"
        )
    lines.extend(
        [
            f"| **Total** | **{total:,}** | **100.00%** | "
            f"**{class_totals['1']:,}** | **{class_totals['0']:,}** | — | — |",
            "",
            "## Verification",
            "",
            "- Image IDs are unique in the input and assigned once.",
            "- Pairwise intersections of the train, validation, and test ID sets are empty.",
            f"- Pneumonia total: {class_totals['1']:,}; Healthy total: {class_totals['0']:,}.",
            "- The CSV stores one split assignment per image; no model-specific split is generated.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8", newline="\n")


def create_split(project_root: Path) -> dict[str, Any]:
    rows = _read_input(project_root / INPUT_PATH)
    assignments = _make_assignments(rows)
    _validate_no_overlap(rows, assignments)

    output_path = project_root / OUTPUT_PATH
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "image_id": row["image_id"],
                    "image_path": row["image_path"],
                    "label": row["binary_label"],
                    "split": assignments[row["image_id"]],
                    "x": row["x"],
                    "y": row["y"],
                    "width": row["width"],
                    "height": row["height"],
                }
            )

    stats_path = project_root / STATS_PATH
    _write_stats(stats_path, rows, assignments)
    split_totals = Counter(assignments.values())
    class_by_split: dict[str, Counter[str]] = {
        split: Counter() for split, _ in SPLIT_RATIOS
    }
    for row in rows:
        class_by_split[assignments[row["image_id"]]][row["binary_label"]] += 1
    return {
        "rows": len(rows),
        "seed": SEED,
        "split_counts": dict(split_totals),
        "class_counts": {
            split: dict(class_by_split[split]) for split, _ in SPLIT_RATIOS
        },
        "overlap_verified": True,
        "output": OUTPUT_PATH.as_posix(),
        "stats": STATS_PATH.as_posix(),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Create the fixed seed-42 stratified dataset split."
    )
    parser.parse_args(argv)
    project_root = Path(__file__).resolve().parents[1]
    try:
        result = create_split(project_root)
    except (OSError, csv.Error, ValueError, KeyError) as error:
        print(f"Split generation failed: {error}", file=sys.stderr)
        return 1
    print("Split generation complete")
    for key, value in result.items():
        print(f"{key}={value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
