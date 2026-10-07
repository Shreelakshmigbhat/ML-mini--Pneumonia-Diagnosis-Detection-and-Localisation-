"""Fit train-only normalization statistics and verify sample preprocessing."""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import PneumoniaDataset, SPLIT_METADATA
from src.data.preprocessing import (
    DEFAULT_STATS_PATH,
    load_normalization_stats,
    read_dicom_pixels,
    resize_and_scale_pixels,
    save_normalization_stats,
    fit_training_stats_from_dataset,
)


def _read_source_metadata(project_root: Path) -> dict[str, dict[str, str]]:
    source = project_root / "metadata" / "pneumonia_metadata.csv"
    with source.open("r", newline="", encoding="utf-8-sig") as source_file:
        rows = list(csv.DictReader(source_file))
    return {row["image_id"]: row for row in rows}


def _save_example(sample: dict[str, object], output_path: Path) -> None:
    image_path = sample["image_path"]
    assert isinstance(image_path, Path)
    pixels, pixel_scale = read_dicom_pixels(image_path)
    display = resize_and_scale_pixels(pixels, pixel_scale)
    canvas = Image.fromarray(np.round(display * 255).astype(np.uint8), mode="L")
    draw = ImageDraw.Draw(canvas)
    original_height, original_width = pixels.shape
    for x, y, width, height in sample["boxes"]:
        left = x * canvas.width / original_width
        top = y * canvas.height / original_height
        right = (x + width) * canvas.width / original_width
        bottom = (y + height) * canvas.height / original_height
        draw.rectangle((left, top, right, bottom), outline=255, width=1)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def verify(samples: int, stats_path: Path) -> int:
    if samples <= 0:
        raise ValueError("--samples must be a positive integer")

    root = PROJECT_ROOT
    train_dataset = PneumoniaDataset(
        root, "train", _placeholder_stats(), SPLIT_METADATA
    )
    if stats_path.exists():
        stats = load_normalization_stats(stats_path)
        print(f"Loaded normalization statistics: {stats_path}")
    else:
        print(
            f"Computing mean/std from {len(train_dataset):,} train images only; "
            "this streams images without caching copies."
        )
        stats = fit_training_stats_from_dataset(train_dataset, stats_path)
        print(f"Saved normalization statistics: {stats_path}")

    datasets = {
        split: PneumoniaDataset(root, split, stats, SPLIT_METADATA)
        for split in ("train", "validation", "test")
    }
    all_records = [
        (dataset, index)
        for dataset in datasets.values()
        for index in range(len(dataset))
    ]
    chosen = random.Random(42).sample(all_records, min(samples, len(all_records)))
    source_records = _read_source_metadata(root)
    output_dir = root / "metadata" / "preprocessing_examples"

    for index, (dataset, row_index) in enumerate(chosen, start=1):
        sample = dataset[row_index]
        image = sample["image"]
        assert isinstance(image, np.ndarray)
        assert image.shape == (128, 128), image.shape
        assert image.dtype == np.float32, image.dtype
        row = dataset.records[row_index]
        source_row = source_records.get(sample["image_id"])
        if source_row is None:
            raise AssertionError(f"No Stage 1 row for {sample['image_id']}")
        if int(source_row["binary_label"]) != sample["label"]:
            raise AssertionError(f"Label mismatch for {sample['image_id']}")
        if source_row["image_path"] != row["image_path"]:
            raise AssertionError(f"Image path mismatch for {sample['image_id']}")
        for column in ("x", "y", "width", "height"):
            if json.loads(source_row[column]) != json.loads(row[column]):
                raise AssertionError(
                    f"{column} box data mismatch for {sample['image_id']}"
                )
        if bool(sample["boxes"].shape[0]) != (sample["label"] == 1):
            raise AssertionError(f"Label/box association mismatch for {sample['image_id']}")

        filename = f"{index:02d}_{sample['image_id']}.png"
        _save_example(sample, output_dir / filename)
        print(
            f"{sample['image_id']} split={sample['split']} "
            f"label={sample['label']} boxes={sample['boxes'].shape[0]} "
            f"shape={image.shape} dtype={image.dtype} saved={filename}"
        )

    print(
        f"Verified {len(chosen)} deterministic random images; "
        f"mean={stats.mean:.8f}, std={stats.std:.8f}; "
        f"examples={output_dir.relative_to(root).as_posix()}"
    )
    return 0


def _placeholder_stats():
    # A train-only source marker is needed to construct a dataset before fitting.
    from src.data.preprocessing import NormalizationStats

    return NormalizationStats(mean=0.0, std=1.0, pixel_scale=1.0)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Verify dynamic 128x128 DICOM preprocessing and metadata links."
    )
    parser.add_argument("--samples", type=int, default=6)
    parser.add_argument(
        "--stats",
        type=Path,
        default=PROJECT_ROOT / DEFAULT_STATS_PATH,
        help="Path where train-only normalization statistics are stored.",
    )
    args = parser.parse_args(argv)
    try:
        return verify(args.samples, args.stats)
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
        print(f"Preprocessing verification failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
