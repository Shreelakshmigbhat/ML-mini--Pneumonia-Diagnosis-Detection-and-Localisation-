"""Checkpoint-backed CAM inference and box-level localization evaluation."""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pydicom
import torch
from PIL import Image, ImageDraw

from src.data.dataset import SPLIT_METADATA
from src.data.preprocessing import (
    DEFAULT_STATS_PATH,
    NormalizationStats,
    load_normalization_stats,
    preprocess_image,
    read_dicom_pixels,
)
from src.localization.bounding_box import (
    Box,
    activated_mask,
    clusters_to_boxes,
)
from src.localization.cam import compute_cam
from src.localization.clustering import find_clusters
from src.localization.iou import image_iou
from src.models.cnn import INPUT_SIZE, Custom10LayerCNN, count_parameters

CHECKPOINT_PATH = Path("models") / "cnn_best.pth"
RESULT_PATH = Path("results") / "cam_results.json"
LABELS_PATH = Path("Data") / "stage_2_train_labels.csv"
REFERENCE_CAM_IOU = 0.1508
PNEUMONIA_CLASS_INDEX = 1


@dataclass(frozen=True)
class LocalizationConfig:
    """Reproducible CAM and post-processing settings."""

    threshold: float = 0.5
    apply_two_sigma_filter: bool = True
    batch_size: int = 32
    example_count: int = 5

    def __post_init__(self) -> None:
        if not 0.0 < self.threshold <= 1.0:
            raise ValueError("threshold must be greater than 0 and at most 1")
        if self.batch_size <= 0 or self.example_count < 5:
            raise ValueError("batch_size must be positive and example_count at least 5")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_cnn_checkpoint(
    project_root: Path,
    device: torch.device | None = None,
) -> tuple[Custom10LayerCNN, torch.device, dict[str, Any]]:
    """Load and validate the best Stage 8 checkpoint without training."""
    root = project_root.resolve()
    checkpoint_path = root / CHECKPOINT_PATH
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Stage 8 CNN checkpoint is required before CAM: {checkpoint_path}"
        )
    selected_device = device or torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "xpu"
        if hasattr(torch, "xpu") and torch.xpu.is_available()
        else "cpu"
    )
    checkpoint = torch.load(
        checkpoint_path, map_location=selected_device, weights_only=True
    )
    if not isinstance(checkpoint, dict):
        raise ValueError("CNN checkpoint must contain the Stage 8 checkpoint mapping")

    split_path = root / SPLIT_METADATA
    if checkpoint.get("split_metadata_sha256") != _sha256(split_path):
        raise ValueError("CNN checkpoint was trained with a different data split")

    model = Custom10LayerCNN().to(selected_device)
    if checkpoint.get("architecture") != model.architecture_config():
        raise ValueError("CNN checkpoint architecture does not match Stage 8")
    state = checkpoint.get("state_dict")
    if not isinstance(state, dict):
        raise ValueError("CNN checkpoint is missing its state_dict")
    model.load_state_dict(state, strict=True)
    model.eval()

    compatibility = model.verify_cam_compatibility()
    if not compatibility["cam_compatible"]:
        raise ValueError("Loaded CNN failed its CAM compatibility check")
    if model.cam_weights().shape[0] <= PNEUMONIA_CLASS_INDEX:
        raise ValueError("CNN checkpoint has no Pneumonia classifier weight row")
    return model, selected_device, {
        "checkpoint_path": CHECKPOINT_PATH.as_posix(),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "parameter_count": count_parameters(model),
        "architecture": model.architecture_config(),
        "cam_compatibility": compatibility,
        "split_metadata_sha256": checkpoint["split_metadata_sha256"],
    }


def _read_test_positive_images(project_root: Path) -> list[dict[str, str]]:
    split_path = project_root / SPLIT_METADATA
    with split_path.open("r", newline="", encoding="utf-8-sig") as stream:
        records = list(csv.DictReader(stream))
    if not records:
        raise ValueError(f"No records in fixed split metadata: {split_path}")
    seen: set[str] = set()
    positives: list[dict[str, str]] = []
    for row in records:
        if row["split"] != "test" or row["label"] != "1":
            continue
        image_id = row["image_id"].strip()
        key = image_id.casefold()
        if not image_id or key in seen:
            raise ValueError(f"Empty or duplicate test image ID: {image_id!r}")
        seen.add(key)
        positives.append(
            {"image_id": image_id, "image_path": row["image_path"]}
        )
    if not positives:
        raise ValueError("Fixed test split contains no Pneumonia-positive images")
    return positives


def _read_source_ground_truth(
    project_root: Path, expected_ids: set[str]
) -> dict[str, list[Box]]:
    """Read positive boxes from the original labels CSV for evaluation only."""
    labels_path = project_root / LABELS_PATH
    ground_truth: dict[str, list[Box]] = {}
    with labels_path.open("r", newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        required = {"patientId", "x", "y", "width", "height", "Target"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(f"RSNA labels CSV is missing required fields: {labels_path}")
        for row in reader:
            image_id = row["patientId"].strip()
            if image_id not in expected_ids or row["Target"] != "1":
                continue
            box = tuple(
                float(row[column]) for column in ("x", "y", "width", "height")
            )
            if not np.isfinite(box).all() or box[2] <= 0 or box[3] <= 0:
                raise ValueError(f"Invalid source ground-truth box for {image_id}")
            ground_truth.setdefault(image_id, []).append(box)  # type: ignore[arg-type]
    missing = expected_ids - ground_truth.keys()
    if missing:
        sample = ", ".join(sorted(missing)[:5])
        raise ValueError(f"Test Pneumonia IDs have no positive source boxes: {sample}")
    return ground_truth


def _dicom_dimensions(path: Path) -> tuple[int, int]:
    header = pydicom.dcmread(
        path, stop_before_pixels=True, specific_tags=("Rows", "Columns")
    )
    height, width = int(header.Rows), int(header.Columns)
    if height <= 0 or width <= 0:
        raise ValueError(f"Invalid image dimensions in DICOM header: {path}")
    return height, width


def _safe_image_path(project_root: Path, relative_path: str) -> Path:
    path = (project_root / relative_path).resolve()
    if not path.is_relative_to(project_root.resolve()):
        raise ValueError(f"Image path escapes project root: {relative_path}")
    if not path.is_file():
        raise FileNotFoundError(f"Test image is missing: {path}")
    return path


def _generate_cams(
    project_root: Path,
    model: Custom10LayerCNN,
    device: torch.device,
    stats: NormalizationStats,
    positives: list[dict[str, str]],
    batch_size: int,
) -> dict[str, dict[str, Any]]:
    """Run class-1 CAM inference from image IDs/paths only; no boxes are passed."""
    model.eval()
    cam_outputs: dict[str, dict[str, Any]] = {}
    for offset in range(0, len(positives), batch_size):
        batch_records = positives[offset : offset + batch_size]
        paths = [
            _safe_image_path(project_root, record["image_path"])
            for record in batch_records
        ]
        processed = np.stack([preprocess_image(path, stats) for path in paths])
        images = torch.from_numpy(processed).unsqueeze(1).to(device)
        cams = compute_cam(
            model,
            images,
            class_index=PNEUMONIA_CLASS_INDEX,
            output_size=INPUT_SIZE,
        ).cpu().numpy()
        for record, path, cam in zip(batch_records, paths, cams):
            cam_outputs[record["image_id"]] = {
                "path": path,
                "cam": np.asarray(cam, dtype=np.float32),
                "image_size": _dicom_dimensions(path),
            }
        processed_count = min(offset + len(batch_records), len(positives))
        if processed_count % 100 < batch_size or processed_count == len(positives):
            print(f"Generated CAMs: {processed_count:,}/{len(positives):,}", flush=True)
    return cam_outputs


def _make_predictions(
    cam_outputs: dict[str, dict[str, Any]],
    config: LocalizationConfig,
) -> dict[str, list[Box]]:
    predictions: dict[str, list[Box]] = {}
    for image_id, item in cam_outputs.items():
        mask = activated_mask(item["cam"], config.threshold)
        clusters = find_clusters(mask)
        predictions[image_id] = clusters_to_boxes(
            clusters,
            map_size=item["cam"].shape,
            image_size=item["image_size"],
            apply_two_sigma=config.apply_two_sigma_filter,
        )
    return predictions


def _heatmap_rgb(cam: np.ndarray, size: tuple[int, int]) -> Image.Image:
    grayscale = Image.fromarray(np.uint8(np.clip(cam, 0.0, 1.0) * 255))
    resized = np.asarray(grayscale.resize(size, Image.Resampling.BILINEAR)) / 255.0
    red = np.clip(1.5 * resized, 0.0, 1.0)
    green = np.clip(1.5 - 1.5 * np.abs(2.0 * resized - 1.0), 0.0, 1.0)
    blue = np.clip(1.5 * (1.0 - resized), 0.0, 1.0)
    rgb = np.stack((red, green, blue), axis=-1)
    return Image.fromarray(np.uint8(rgb * 255), mode="RGB")


def _draw_box_set(
    canvas: Image.Image,
    boxes: list[Box],
    *,
    color: tuple[int, int, int],
    image_size: tuple[int, int],
) -> None:
    draw = ImageDraw.Draw(canvas)
    image_height, image_width = image_size
    scale_x = canvas.width / image_width
    scale_y = canvas.height / image_height
    for x, y, width, height in boxes:
        draw.rectangle(
            (
                x * scale_x,
                y * scale_y,
                (x + width) * scale_x,
                (y + height) * scale_y,
            ),
            outline=color,
            width=max(2, canvas.width // 256),
        )


def _visualize_example(
    project_root: Path,
    image_id: str,
    item: dict[str, Any],
    ground_truth: list[Box],
    predictions: list[Box],
    score: float,
    output_path: Path,
) -> None:
    pixels, pixel_scale = read_dicom_pixels(item["path"])
    original_array = np.uint8(np.clip(pixels / pixel_scale, 0.0, 1.0) * 255)
    original = Image.fromarray(original_array, mode="L").convert("RGB")
    size = original.size
    original_for_gt = original.copy()
    original_for_prediction = original.copy()
    _draw_box_set(original_for_gt, ground_truth, color=(0, 255, 0), image_size=(size[1], size[0]))
    _draw_box_set(
        original_for_prediction,
        predictions,
        color=(255, 40, 40),
        image_size=(size[1], size[0]),
    )
    heatmap = _heatmap_rgb(item["cam"], size)
    cam_overlay = Image.blend(original, heatmap, alpha=0.45)

    panel_size = (384, 384)
    labels = (
        ("Original", original),
        ("Ground truth", original_for_gt),
        ("CAM overlay", cam_overlay),
        ("Predicted boxes", original_for_prediction),
    )
    sheet = Image.new("RGB", (panel_size[0] * 2, panel_size[1] * 2), "white")
    draw = ImageDraw.Draw(sheet)
    for index, (label, panel) in enumerate(labels):
        x0 = (index % 2) * panel_size[0]
        y0 = (index // 2) * panel_size[1]
        resized = panel.resize(panel_size, Image.Resampling.BILINEAR)
        sheet.paste(resized, (x0, y0))
        draw.rectangle((x0, y0, x0 + panel_size[0], y0 + 24), fill=(0, 0, 0))
        draw.text((x0 + 8, y0 + 5), label, fill=(255, 255, 255))
    draw.text((8, 2 * panel_size[1] - 18), f"{image_id} | IoU {score:.4f}", fill=(0, 0, 0))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path)


def _select_example_ids(scores: dict[str, float], count: int) -> list[str]:
    ordered = sorted(scores, key=lambda image_id: (scores[image_id], image_id))
    if len(ordered) <= count:
        return ordered
    positions = np.linspace(0, len(ordered) - 1, count, dtype=int)
    return [ordered[int(position)] for position in positions]


def run_localization(
    project_root: Path,
    config: LocalizationConfig | None = None,
) -> dict[str, Any]:
    """Generate test-set CAMs, boxes, visual examples, and localization metrics."""
    root = project_root.resolve()
    config = config or LocalizationConfig()
    model, device, checkpoint_info = load_cnn_checkpoint(root)
    stats = load_normalization_stats(root / DEFAULT_STATS_PATH)
    if stats.source_split != "train" or stats.image_size != INPUT_SIZE:
        raise ValueError("CAM input must use the existing train-fitted 128x128 stats")

    positives = _read_test_positive_images(root)
    positive_ids = {record["image_id"] for record in positives}
    cam_outputs = _generate_cams(
        root, model, device, stats, positives, config.batch_size
    )
    predictions = _make_predictions(cam_outputs, config)
    ground_truth = _read_source_ground_truth(root, positive_ids)
    if set(cam_outputs) != set(ground_truth) or set(predictions) != set(ground_truth):
        raise AssertionError("CAM, prediction, and source annotation IDs do not match")

    scores = {
        image_id: image_iou(predictions[image_id], ground_truth[image_id])
        for image_id in sorted(ground_truth)
    }
    values = np.asarray(list(scores.values()), dtype=np.float64)
    if not len(values):
        raise ValueError("No test images were evaluated")
    examples_directory = root / "results" / "cam_examples"
    example_ids = _select_example_ids(scores, max(5, config.example_count))
    for image_id in example_ids:
        score = scores[image_id]
        _visualize_example(
            root,
            image_id,
            cam_outputs[image_id],
            ground_truth[image_id],
            predictions[image_id],
            score,
            examples_directory / f"{image_id}_iou_{score:.4f}.png",
        )

    result: dict[str, Any] = {
        "model": "Stage 8 custom CNN + CAM + DFS clustering",
        "threshold": config.threshold,
        "threshold_rule": "normalized ReLU CAM >= threshold (inclusive)",
        "num_images": int(len(values)),
        "mean_iou": float(values.mean()),
        "median_iou": float(np.median(values)),
        "percentage_iou_gt_0": float(np.mean(values > 0.0) * 100.0),
        "percentage_iou_ge_0_5": float(np.mean(values >= 0.5) * 100.0),
        "num_images_with_no_predicted_boxes": int(
            sum(not predictions[image_id] for image_id in predictions)
        ),
        "two_sigma_filter": {
            "enabled": config.apply_two_sigma_filter,
            "rule": (
                "retain per-image DFS clusters whose activated-pixel count lies "
                "within the mean +/- 2 population standard deviations"
            ),
        },
        "clustering": "iterative 8-connected depth-first search on the 128x128 CAM",
        "iou_matching": (
            "descending-IoU greedy one-to-one box matching; image IoU is matched "
            "IoU sum divided by max(predicted box count, ground-truth box count), "
            "with unmatched boxes contributing zero"
        ),
        "ground_truth_source": LABELS_PATH.as_posix(),
        "ground_truth_used_for_cam": False,
        "cam_class_index": PNEUMONIA_CLASS_INDEX,
        "checkpoint": checkpoint_info,
        "reference_test_iou": REFERENCE_CAM_IOU,
        "difference_from_reference": float(values.mean() - REFERENCE_CAM_IOU),
        "example_files": [
            (Path("results") / "cam_examples" / f"{image_id}_iou_{scores[image_id]:.4f}.png").as_posix()
            for image_id in example_ids
        ],
        "image_results": [
            {
                "image_id": image_id,
                "iou": scores[image_id],
                "ground_truth_box_count": len(ground_truth[image_id]),
                "predicted_box_count": len(predictions[image_id]),
            }
            for image_id in sorted(scores)
        ],
    }
    result_path = root / RESULT_PATH
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(f"Test Pneumonia images evaluated: {result['num_images']:,}")
    print(f"CAM threshold: {config.threshold:.3f}")
    print(f"Mean IoU: {result['mean_iou']:.6f}")
    print(f"Median IoU: {result['median_iou']:.6f}")
    print(f"IoU > 0: {result['percentage_iou_gt_0']:.2f}%")
    print(f"IoU >= 0.5: {result['percentage_iou_ge_0_5']:.2f}%")
    print(f"Reference CAM IoU difference: {result['difference_from_reference']:+.6f}")
    print(f"Saved results: {result_path}")
    print(f"Saved visual examples: {examples_directory}")
    return result
