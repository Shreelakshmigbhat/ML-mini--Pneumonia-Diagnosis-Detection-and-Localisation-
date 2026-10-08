"""Run Stage 13 Grad-CAM++ localization independently of the Phase 1 pipeline."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.preprocessing import (  # noqa: E402
    DEFAULT_STATS_PATH,
    NormalizationStats,
    load_normalization_stats,
    preprocess_image,
    read_dicom_pixels,
)
from src.localization.bounding_box import (  # noqa: E402
    Box,
    activated_mask,
    clusters_to_boxes,
)
from src.localization.cam import compute_cam  # noqa: E402
from src.localization.clustering import find_clusters  # noqa: E402
from src.localization.improved_cam import compute_gradcam_plus_plus  # noqa: E402
from src.localization.iou import image_iou  # noqa: E402
from src.localization.pipeline import (  # noqa: E402
    LocalizationConfig,
    _dicom_dimensions,
    _read_source_ground_truth,
    _read_test_positive_images,
    _safe_image_path,
    load_cnn_checkpoint,
)
from src.models.cnn import INPUT_SIZE, Custom10LayerCNN  # noqa: E402


BASELINE_PATH = Path("results") / "cam_results.json"
RESULT_PATH = Path("results") / "phase2_improved_cam_results.json"
COMPARISON_PATH = Path("results") / "phase2_baseline_vs_improved.csv"
EXAMPLES_DIRECTORY = Path("results") / "phase2_improved_examples"
DEFAULT_THRESHOLD = 0.5
DEFAULT_BATCH_SIZE = 16
SMOKE_COUNT = 12
EXAMPLE_COUNT = 10


def _prepare_batch(
    root: Path,
    stats: NormalizationStats,
    records: list[dict[str, str]],
    device: torch.device,
) -> tuple[torch.Tensor, list[Path]]:
    paths = [
        _safe_image_path(root, record["image_path"]) for record in records
    ]
    processed = np.stack([preprocess_image(path, stats) for path in paths])
    images = torch.from_numpy(processed).unsqueeze(1).to(device)
    return images, paths


def _predictions_from_maps(
    image_ids: list[str],
    maps: np.ndarray,
    image_sizes: list[tuple[int, int]],
    threshold: float,
    apply_two_sigma_filter: bool,
) -> tuple[
    dict[str, list[Box]],
    dict[str, int],
    dict[str, np.ndarray],
]:
    predictions: dict[str, list[Box]] = {}
    component_counts: dict[str, int] = {}
    masks: dict[str, np.ndarray] = {}
    for image_id, cam_map, image_size in zip(image_ids, maps, image_sizes):
        cam_map = np.asarray(cam_map, dtype=np.float32)
        if cam_map.shape != INPUT_SIZE:
            raise ValueError(f"Unexpected Grad-CAM++ shape for {image_id}: {cam_map.shape}")
        if not np.isfinite(cam_map).all():
            raise FloatingPointError(f"Non-finite Grad-CAM++ map for {image_id}")
        mask = activated_mask(cam_map, threshold)
        clusters = find_clusters(mask)
        boxes = clusters_to_boxes(
            clusters,
            map_size=cam_map.shape,
            image_size=image_size,
            apply_two_sigma=apply_two_sigma_filter,
        )
        height, width = image_size
        for x, y, box_width, box_height in boxes:
            if not np.isfinite((x, y, box_width, box_height)).all():
                raise FloatingPointError(f"Non-finite predicted box for {image_id}")
            if (
                x < 0
                or y < 0
                or box_width <= 0
                or box_height <= 0
                or x + box_width > width + 1e-6
                or y + box_height > height + 1e-6
            ):
                raise ValueError(f"Predicted box is outside image bounds for {image_id}")
        predictions[image_id] = boxes
        component_counts[image_id] = len(clusters)
        masks[image_id] = mask
    return predictions, component_counts, masks


def _heatmap_rgb(cam: np.ndarray, size: tuple[int, int]) -> Image.Image:
    grayscale = Image.fromarray(
        np.uint8(np.clip(cam, 0.0, 1.0) * 255), mode="L"
    )
    resized = np.asarray(
        grayscale.resize(size, Image.Resampling.BILINEAR), dtype=np.float32
    ) / 255.0
    red = np.clip(1.5 * resized, 0.0, 1.0)
    green = np.clip(1.5 - 1.5 * np.abs(2.0 * resized - 1.0), 0.0, 1.0)
    blue = np.clip(1.5 * (1.0 - resized), 0.0, 1.0)
    rgb = np.stack((red, green, blue), axis=-1)
    return Image.fromarray(np.uint8(rgb * 255), mode="RGB")


def _draw_boxes(
    image: Image.Image,
    boxes: list[Box],
    *,
    image_size: tuple[int, int],
    color: tuple[int, int, int],
) -> None:
    height, width = image_size
    scale_x, scale_y = image.width / width, image.height / height
    draw = ImageDraw.Draw(image)
    for x, y, box_width, box_height in boxes:
        draw.rectangle(
            (
                x * scale_x,
                y * scale_y,
                (x + box_width) * scale_x,
                (y + box_height) * scale_y,
            ),
            outline=color,
            width=max(2, image.width // 256),
        )


def _read_original(path: Path) -> Image.Image:
    pixels, pixel_scale = read_dicom_pixels(path)
    values = np.uint8(np.clip(pixels / pixel_scale, 0.0, 1.0) * 255)
    return Image.fromarray(values, mode="L").convert("RGB")


def _make_comparison_panel(
    image_id: str,
    path: Path,
    baseline_map: np.ndarray,
    improved_map: np.ndarray,
    ground_truth: list[Box],
    baseline_boxes: list[Box],
    improved_boxes: list[Box],
    improved_iou: float,
    baseline_iou: float,
    output_path: Path,
) -> None:
    original = _read_original(path)
    image_size = (original.height, original.width)
    panel_size = (384, 384)
    baseline_overlay = Image.blend(
        original, _heatmap_rgb(baseline_map, original.size), alpha=0.45
    )
    improved_overlay = Image.blend(
        original, _heatmap_rgb(improved_map, original.size), alpha=0.45
    )
    baseline_panel = baseline_overlay.copy()
    _draw_boxes(
        baseline_panel,
        baseline_boxes,
        image_size=image_size,
        color=(255, 50, 50),
    )
    improved_panel = improved_overlay.copy()
    _draw_boxes(
        improved_panel,
        improved_boxes,
        image_size=image_size,
        color=(255, 50, 50),
    )
    ground_truth_panel = original.copy()
    _draw_boxes(
        ground_truth_panel,
        ground_truth,
        image_size=image_size,
        color=(0, 220, 0),
    )
    panels = (
        ("A. Original X-ray", original),
        ("B. Phase 1 CAM + boxes", baseline_panel),
        ("C. Phase 2 Grad-CAM++ + boxes", improved_panel),
        ("D. Ground truth", ground_truth_panel),
    )
    sheet = Image.new("RGB", (panel_size[0] * 2, panel_size[1] * 2), "white")
    draw = ImageDraw.Draw(sheet)
    for index, (label, panel) in enumerate(panels):
        x0 = index % 2 * panel_size[0]
        y0 = index // 2 * panel_size[1]
        sheet.paste(panel.resize(panel_size, Image.Resampling.BILINEAR), (x0, y0))
        draw.rectangle(
            (x0, y0, x0 + panel_size[0], y0 + 24), fill=(0, 0, 0)
        )
        draw.text((x0 + 8, y0 + 5), label, fill=(255, 255, 255))
    draw.text(
        (8, panel_size[1] * 2 - 18),
        f"{image_id} | baseline IoU={baseline_iou:.4f} | improved IoU={improved_iou:.4f}",
        fill=(0, 0, 0),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path)


def _score_map(
    predictions: dict[str, list[Box]],
    ground_truth: dict[str, list[Box]],
) -> dict[str, float]:
    if set(predictions) != set(ground_truth):
        raise AssertionError("Prediction IDs do not match fixed test annotations")
    return {
        image_id: image_iou(predictions[image_id], ground_truth[image_id])
        for image_id in sorted(ground_truth)
    }


def _select_comparison_examples(
    baseline_scores: dict[str, float],
    improved_scores: dict[str, float],
    count: int,
) -> list[str]:
    common_ids = sorted(set(baseline_scores) & set(improved_scores))
    deltas = {
        image_id: improved_scores[image_id] - baseline_scores[image_id]
        for image_id in common_ids
    }
    selected: list[str] = []
    improved = sorted(
        (image_id for image_id in common_ids if deltas[image_id] > 1e-12),
        key=lambda image_id: (-deltas[image_id], image_id),
    )
    unchanged = sorted(
        image_id for image_id in common_ids if abs(deltas[image_id]) <= 1e-12
    )
    worsened = sorted(
        (image_id for image_id in common_ids if deltas[image_id] < -1e-12),
        key=lambda image_id: (deltas[image_id], image_id),
    )
    for group in (improved, unchanged, worsened):
        if group:
            selected.append(group[0])
    by_delta = sorted(
        common_ids, key=lambda image_id: (deltas[image_id], image_id)
    )
    if count > len(selected):
        slots = count - len(selected)
        positions = np.linspace(0, len(by_delta) - 1, slots, dtype=int)
        for position in positions:
            image_id = by_delta[int(position)]
            if image_id not in selected:
                selected.append(image_id)
            if len(selected) == count:
                break
    for image_id in by_delta:
        if len(selected) >= min(count, len(common_ids)):
            break
        if image_id not in selected:
            selected.append(image_id)
    return selected[:count]


def _verify_baseline(
    saved: dict[str, Any], scores: dict[str, float]
) -> dict[str, float]:
    if saved.get("threshold") != 0.5 or int(saved.get("num_images", -1)) != len(scores):
        raise ValueError("Phase 1 result is not the expected fixed test baseline")
    stored = {row["image_id"]: row for row in saved["image_results"]}
    if set(stored) != set(scores):
        raise ValueError("Phase 1 saved image IDs differ from fixed test IDs")
    for image_id, score in scores.items():
        if abs(float(stored[image_id]["iou"]) - score) > 1e-12:
            raise ValueError(f"Recomputed Phase 1 IoU differs for {image_id}")
    values = np.asarray(list(scores.values()), dtype=np.float64)
    metrics = {
        "mean_iou": float(values.mean()),
        "median_iou": float(np.median(values)),
        "percentage_iou_gt_0": float(np.mean(values > 0.0) * 100.0),
        "percentage_iou_ge_0_5": float(np.mean(values >= 0.5) * 100.0),
    }
    for field, value in metrics.items():
        if abs(value - float(saved[field])) > 1e-12:
            raise ValueError(f"Recomputed Phase 1 {field} differs from saved baseline")
    return metrics


def _run_images(
    root: Path,
    model: Custom10LayerCNN,
    device: torch.device,
    stats: NormalizationStats,
    records: list[dict[str, str]],
    *,
    threshold: float,
    batch_size: int,
    apply_two_sigma_filter: bool,
    epsilon: float,
) -> tuple[
    dict[str, dict[str, Any]],
    dict[str, list[Box]],
    dict[str, int],
    dict[str, np.ndarray],
]:
    outputs: dict[str, dict[str, Any]] = {}
    all_maps: list[np.ndarray] = []
    all_sizes: list[tuple[int, int]] = []
    image_ids: list[str] = []
    for offset in range(0, len(records), batch_size):
        batch = records[offset : offset + batch_size]
        images, paths = _prepare_batch(root, stats, batch, device)
        maps = compute_gradcam_plus_plus(
            model,
            images,
            class_index=1,
            output_size=INPUT_SIZE,
            epsilon=epsilon,
        ).cpu().numpy()
        if not np.isfinite(maps).all():
            raise FloatingPointError("Grad-CAM++ batch contains non-finite values")
        sizes = [_dicom_dimensions(path) for path in paths]
        for index, (record, path, image_size) in enumerate(
            zip(batch, paths, sizes)
        ):
            image_id = record["image_id"]
            outputs[image_id] = {
                "path": path,
                "image_size": image_size,
                "cam": maps[index].astype(np.float32, copy=False),
            }
            all_maps.append(maps[index].astype(np.float32, copy=False))
            all_sizes.append(image_size)
            image_ids.append(image_id)
        processed = min(offset + len(batch), len(records))
        print(f"Processed {processed:,}/{len(records):,}", flush=True)
    if not records:
        raise ValueError("No test images were selected")
    final_predictions, final_counts, final_masks = _predictions_from_maps(
        image_ids,
        np.stack(all_maps),
        all_sizes,
        threshold,
        apply_two_sigma_filter,
    )
    return outputs, final_predictions, final_counts, final_masks


def _smoke_test(
    root: Path,
    model: Custom10LayerCNN,
    device: torch.device,
    stats: NormalizationStats,
    records: list[dict[str, str]],
    *,
    threshold: float,
    batch_size: int,
    apply_two_sigma_filter: bool,
    epsilon: float,
) -> None:
    smoke_records = sorted(records, key=lambda row: row["image_id"])[:SMOKE_COUNT]
    outputs, predictions, component_counts, masks = _run_images(
        root,
        model,
        device,
        stats,
        smoke_records,
        threshold=threshold,
        batch_size=batch_size,
        apply_two_sigma_filter=apply_two_sigma_filter,
        epsilon=epsilon,
    )
    for image_id, item in outputs.items():
        cam_map = item["cam"]
        mask = masks[image_id]
        height, width = item["image_size"]
        if cam_map.shape != INPUT_SIZE or not np.isfinite(cam_map).all():
            raise AssertionError(f"Invalid CAM for smoke-test image {image_id}")
        if mask.shape != INPUT_SIZE or mask.dtype != np.bool_:
            raise AssertionError(f"Invalid activation mask for smoke-test image {image_id}")
        if component_counts[image_id] < 0:
            raise AssertionError(f"Invalid component count for {image_id}")
        for x, y, box_width, box_height in predictions[image_id]:
            if (
                x < 0
                or y < 0
                or box_width <= 0
                or box_height <= 0
                or x + box_width > width + 1e-6
                or y + box_height > height + 1e-6
            ):
                raise AssertionError(f"Invalid smoke-test box for {image_id}")

    preview_id = next(iter(outputs))
    preview = outputs[preview_id]
    original = _read_original(preview["path"])
    overlay = Image.blend(
        original,
        _heatmap_rgb(preview["cam"], original.size),
        alpha=0.45,
    )
    _draw_boxes(
        overlay,
        predictions[preview_id],
        image_size=preview["image_size"],
        color=(255, 50, 50),
    )
    preview_path = root / EXAMPLES_DIRECTORY / "smoke_test_alignment.png"
    preview_path.parent.mkdir(parents=True, exist_ok=True)
    overlay.save(preview_path)
    if Image.open(preview_path).size != original.size:
        raise AssertionError("Smoke-test overlay dimensions do not match source X-ray")
    print(
        f"Smoke test passed: {len(smoke_records)} deterministic images; "
        "CAMs, masks, components, boxes, finite values, bounds, and alignment verified."
    )
    print(f"Smoke alignment preview: {preview_path}")


def _evaluate(
    args: argparse.Namespace,
) -> dict[str, Any]:
    root = PROJECT_ROOT.resolve()
    baseline_path = root / BASELINE_PATH
    if not baseline_path.is_file():
        raise FileNotFoundError(f"Saved Phase 1 result required: {baseline_path}")
    baseline_saved = json.loads(baseline_path.read_text(encoding="utf-8"))
    model, device, checkpoint_info = load_cnn_checkpoint(root)
    baseline_checkpoint_sha256 = baseline_saved.get("checkpoint", {}).get(
        "checkpoint_sha256"
    )
    if checkpoint_info["checkpoint_sha256"] != baseline_checkpoint_sha256:
        raise ValueError("Loaded CNN checkpoint differs from the Phase 1 CAM baseline")
    stats = load_normalization_stats(root / DEFAULT_STATS_PATH)
    if stats.source_split != "train" or stats.image_size != INPUT_SIZE:
        raise ValueError("Expected the existing train-fitted 128x128 preprocessing stats")
    records = _read_test_positive_images(root)
    if int(baseline_saved["num_images"]) != len(records):
        raise ValueError("Fixed test-positive count differs from the Phase 1 baseline")

    if args.smoke_only:
        _smoke_test(
            root,
            model,
            device,
            stats,
            records,
            threshold=args.threshold,
            batch_size=args.batch_size,
            apply_two_sigma_filter=not args.disable_two_sigma_filter,
            epsilon=args.epsilon,
        )
        return {}

    outputs, predictions, _component_counts, _masks = _run_images(
        root,
        model,
        device,
        stats,
        records,
        threshold=args.threshold,
        batch_size=args.batch_size,
        apply_two_sigma_filter=not args.disable_two_sigma_filter,
        epsilon=args.epsilon,
    )
    positive_ids = {record["image_id"] for record in records}
    ground_truth = _read_source_ground_truth(root, positive_ids)
    improved_scores = _score_map(predictions, ground_truth)
    baseline_scores = {
        row["image_id"]: float(row["iou"])
        for row in baseline_saved["image_results"]
    }
    baseline_metrics = _verify_baseline(baseline_saved, baseline_scores)
    improved_values = np.asarray(list(improved_scores.values()), dtype=np.float64)
    if not len(improved_values):
        raise ValueError("No images were evaluated")
    improved_metrics = {
        "mean_iou": float(improved_values.mean()),
        "median_iou": float(np.median(improved_values)),
        "percentage_iou_gt_0": float(np.mean(improved_values > 0.0) * 100.0),
        "percentage_iou_ge_0_5": float(
            np.mean(improved_values >= 0.5) * 100.0
        ),
        "num_images": int(len(improved_values)),
    }
    absolute_improvement = (
        improved_metrics["mean_iou"] - baseline_metrics["mean_iou"]
    )
    relative_improvement = (
        absolute_improvement / baseline_metrics["mean_iou"] * 100.0
    )

    example_ids = _select_comparison_examples(
        baseline_scores, improved_scores, EXAMPLE_COUNT
    )
    examples_dir = root / EXAMPLES_DIRECTORY
    examples_dir.mkdir(parents=True, exist_ok=True)
    with (examples_dir / "examples.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=(
                "image_id",
                "baseline_iou",
                "improved_iou",
                "iou_delta",
                "case_type",
                "visualization",
            ),
        )
        writer.writeheader()
        for index, image_id in enumerate(example_ids, start=1):
            record = next(row for row in records if row["image_id"] == image_id)
            item = outputs[image_id]
            image_tensor, _ = _prepare_batch(root, stats, [record], device)
            with torch.no_grad():
                baseline_map = compute_cam(
                    model,
                    image_tensor,
                    class_index=1,
                    output_size=INPUT_SIZE,
                )[0].cpu().numpy()
            output_name = f"{index:02d}_{image_id}.png"
            relative_path = (EXAMPLES_DIRECTORY / output_name).as_posix()
            _make_comparison_panel(
                image_id,
                item["path"],
                baseline_map,
                item["cam"],
                ground_truth[image_id],
                _baseline_boxes_for_image(baseline_map, item["image_size"]),
                predictions[image_id],
                improved_scores[image_id],
                baseline_scores[image_id],
                root / relative_path,
            )
            delta = improved_scores[image_id] - baseline_scores[image_id]
            case_type = (
                "improved"
                if delta > 1e-12
                else "unchanged"
                if abs(delta) <= 1e-12
                else "worsened"
            )
            writer.writerow(
                {
                    "image_id": image_id,
                    "baseline_iou": baseline_scores[image_id],
                    "improved_iou": improved_scores[image_id],
                    "iou_delta": delta,
                    "case_type": case_type,
                    "visualization": relative_path,
                }
            )

    result = {
        "method_name": "Stage 13 Grad-CAM++ Localization",
        "baseline_method": baseline_saved["model"],
        "new_method": "Frozen Stage 8 CNN with Grad-CAM++ class-1 map and existing box post-processing",
        "parameters": {
            "class_index": 1,
            "threshold": args.threshold,
            "threshold_rule": "normalized Grad-CAM++ >= threshold (inclusive)",
            "apply_two_sigma_filter": not args.disable_two_sigma_filter,
            "two_sigma_filter_rule": "retain per-image 8-connected clusters within mean +/- 2 population standard deviations of activated-pixel count",
            "connectivity": 8,
            "batch_size": args.batch_size,
            "input_size": list(INPUT_SIZE),
            "cam_output_size": list(INPUT_SIZE),
            "epsilon": args.epsilon,
            "smoke_sample_count": SMOKE_COUNT,
            "qualitative_example_count": len(example_ids),
        },
        "test_sample_count": improved_metrics["num_images"],
        **{
            key: value
            for key, value in improved_metrics.items()
            if key != "num_images"
        },
        "baseline_mean_iou": baseline_metrics["mean_iou"],
        "baseline_metrics": baseline_metrics,
        "absolute_improvement": absolute_improvement,
        "relative_improvement_percent": relative_improvement,
        "improved_over_baseline": absolute_improvement > 0.0,
        "stage12_dominant_failure_mode": (
            "Spatial imprecision: diffuse/misplaced activations and oversized predicted boxes."
        ),
        "checkpoint": checkpoint_info,
        "split": "existing fixed test split; 601 Pneumonia-positive images",
        "evaluation_strategy": baseline_saved["iou_matching"],
        "ground_truth_used_for_generation_or_processing": False,
        "visual_example_index": (EXAMPLES_DIRECTORY / "examples.csv").as_posix(),
        "phase1_results_modified": False,
    }
    result_path = root / RESULT_PATH
    result_path.write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    comparison_path = root / COMPARISON_PATH
    with comparison_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=(
                "Method",
                "Mean IoU",
                "Median IoU",
                "IoU > 0",
                "IoU >= 0.5",
            ),
        )
        writer.writeheader()
        writer.writerow(
            {
                "Method": "Phase 1 CAM (threshold 0.50)",
                "Mean IoU": baseline_metrics["mean_iou"],
                "Median IoU": baseline_metrics["median_iou"],
                "IoU > 0": baseline_metrics["percentage_iou_gt_0"],
                "IoU >= 0.5": baseline_metrics["percentage_iou_ge_0_5"],
            }
        )
        writer.writerow(
            {
                "Method": "Phase 2 Grad-CAM++",
                "Mean IoU": improved_metrics["mean_iou"],
                "Median IoU": improved_metrics["median_iou"],
                "IoU > 0": improved_metrics["percentage_iou_gt_0"],
                "IoU >= 0.5": improved_metrics["percentage_iou_ge_0_5"],
            }
        )
    print(json.dumps(result, indent=2))
    print(f"Saved result: {result_path}")
    print(f"Saved comparison: {comparison_path}")
    print(f"Saved {len(example_ids)} qualitative panels: {examples_dir}")
    return result


def _baseline_boxes_for_image(
    baseline_map: np.ndarray,
    image_size: tuple[int, int],
) -> list[Box]:
    config = LocalizationConfig(threshold=0.5)
    mask = activated_mask(baseline_map, config.threshold)
    clusters = find_clusters(mask)
    return clusters_to_boxes(
        clusters,
        map_size=baseline_map.shape,
        image_size=image_size,
        apply_two_sigma=config.apply_two_sigma_filter,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help="Inclusive cutoff on per-image normalized Grad-CAM++ maps (default: 0.5).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Number of fixed-test images processed per inference batch (default: 16).",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=1e-8,
        help="Positive numerical stabilizer used in Grad-CAM++ weighting.",
    )
    parser.add_argument(
        "--disable-two-sigma-filter",
        action="store_true",
        help="Keep every 8-connected activation region.",
    )
    parser.add_argument(
        "--smoke-only",
        action="store_true",
        help="Run the deterministic 12-image validation without full evaluation.",
    )
    args = parser.parse_args()
    if not 0.0 < args.threshold <= 1.0:
        parser.error("--threshold must be greater than 0 and at most 1")
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    if args.epsilon <= 0:
        parser.error("--epsilon must be positive")
    _evaluate(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
