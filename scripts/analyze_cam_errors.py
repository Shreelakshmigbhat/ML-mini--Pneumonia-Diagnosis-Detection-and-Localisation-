"""Analyze Stage 9 CAM thresholds and failures without changing its outputs."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.localization.pipeline import (  # noqa: E402
    LocalizationConfig,
    _generate_cams,
    _make_predictions,
    _read_source_ground_truth,
    _read_test_positive_images,
    _visualize_example,
    load_cnn_checkpoint,
)
from src.data.preprocessing import (  # noqa: E402
    DEFAULT_STATS_PATH,
    load_normalization_stats,
)
from src.localization.iou import image_iou  # noqa: E402
from src.models.cnn import INPUT_SIZE  # noqa: E402


BASELINE_PATH = Path("results") / "cam_results.json"
THRESHOLD_CSV_PATH = Path("results") / "cam_threshold_analysis.csv"
FAILURE_STATS_PATH = Path("results") / "cam_failure_statistics.json"
ERROR_ANALYSIS_DIR = Path("results") / "phase2_error_analysis"
THRESHOLDS = (0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80)


def _box_area(box: tuple[float, float, float, float]) -> float:
    return max(0.0, box[2]) * max(0.0, box[3])


def _baseline_metrics(
    predictions: dict[str, list[tuple[float, float, float, float]]],
    ground_truth: dict[str, list[tuple[float, float, float, float]]],
) -> tuple[dict[str, float], dict[str, int]]:
    scores = {
        image_id: image_iou(predictions[image_id], ground_truth[image_id])
        for image_id in sorted(ground_truth)
    }
    values = np.asarray(list(scores.values()), dtype=np.float64)
    box_counts = {
        image_id: len(predictions[image_id]) for image_id in sorted(predictions)
    }
    return (
        {
            "mean_iou": float(values.mean()),
            "median_iou": float(np.median(values)),
            "percentage_iou_gt_0": float(np.mean(values > 0.0) * 100.0),
            "percentage_iou_ge_0_5": float(np.mean(values >= 0.5) * 100.0),
        },
        box_counts,
    )


def _verify_baseline(
    saved: dict[str, Any],
    scores: dict[str, float],
    box_counts: dict[str, int],
) -> None:
    if saved.get("threshold") != 0.5 or saved.get("num_images") != len(scores):
        raise ValueError("Saved Phase 1 CAM baseline metadata does not match this run")

    stored = {row["image_id"]: row for row in saved["image_results"]}
    if set(stored) != set(scores):
        raise ValueError("Saved Phase 1 image IDs do not match the fixed test split")
    for image_id, score in scores.items():
        if abs(float(stored[image_id]["iou"]) - score) > 1e-12:
            raise ValueError(
                f"Recomputed baseline IoU differs for test image {image_id}"
            )
        if int(stored[image_id]["predicted_box_count"]) != box_counts[image_id]:
            raise ValueError(
                f"Recomputed baseline box count differs for test image {image_id}"
            )

    metrics, _ = _baseline_metrics_from_scores(scores, box_counts)
    for field, actual in metrics.items():
        if abs(float(saved[field]) - actual) > 1e-12:
            raise ValueError(f"Recomputed baseline {field} does not match saved result")


def _baseline_metrics_from_scores(
    scores: dict[str, float], box_counts: dict[str, int]
) -> tuple[dict[str, float], int]:
    del box_counts
    values = np.asarray(list(scores.values()), dtype=np.float64)
    return (
        {
            "mean_iou": float(values.mean()),
            "median_iou": float(np.median(values)),
            "percentage_iou_gt_0": float(np.mean(values > 0.0) * 100.0),
            "percentage_iou_ge_0_5": float(np.mean(values >= 0.5) * 100.0),
        },
        len(values),
    )


def _select_error_examples(
    scores: dict[str, float],
    predictions: dict[str, list[tuple[float, float, float, float]]],
    ground_truth: dict[str, list[tuple[float, float, float, float]]],
    image_sizes: dict[str, tuple[int, int]],
) -> dict[str, list[str]]:
    areas = {
        image_id: sum(_box_area(box) for box in boxes)
        for image_id, boxes in predictions.items()
    }
    image_area = {
        image_id: float(height * width)
        for image_id, (height, width) in image_sizes.items()
    }
    area_fraction = {
        image_id: areas[image_id] / image_area[image_id]
        for image_id in predictions
    }
    assignments: dict[str, list[str]] = {}

    def assign(
        category: str,
        candidates: list[str],
        key: Any,
        *,
        reverse: bool = False,
    ) -> None:
        candidates = sorted(candidates, key=key, reverse=reverse)
        if not candidates:
            return
        candidate = next(
            (image_id for image_id in candidates if image_id not in assignments),
            candidates[0],
        )
        assignments.setdefault(candidate, []).append(category)

    ids = sorted(scores)
    assign(
        "good_localization",
        [image_id for image_id in ids if scores[image_id] >= 0.5],
        key=lambda image_id: scores[image_id],
        reverse=True,
    )
    assign(
        "partial_localization",
        [image_id for image_id in ids if 0.0 < scores[image_id] < 0.5],
        key=lambda image_id: abs(scores[image_id] - 0.25),
    )
    zero_iou_with_boxes = [
        image_id
        for image_id in ids
        if scores[image_id] == 0.0 and predictions[image_id]
    ]
    assign(
        "large_false_positive_region",
        zero_iou_with_boxes,
        key=lambda image_id: area_fraction[image_id],
        reverse=True,
    )
    assign(
        "very_small_predicted_region",
        [image_id for image_id in ids if predictions[image_id]],
        key=lambda image_id: area_fraction[image_id],
    )
    assign(
        "multiple_disconnected_regions",
        [image_id for image_id in ids if len(predictions[image_id]) > 1],
        key=lambda image_id: len(predictions[image_id]),
        reverse=True,
    )
    assign(
        "irrelevant_anatomical_activation",
        zero_iou_with_boxes,
        key=lambda image_id: area_fraction[image_id],
    )
    assign(
        "no_useful_activation_no_predicted_box",
        [image_id for image_id in ids if not predictions[image_id]],
        key=lambda image_id: scores[image_id],
    )
    assign(
        "ground_truth_region_missed",
        [
            image_id
            for image_id in ids
            if scores[image_id] == 0.0
            and predictions[image_id]
            and ground_truth[image_id]
        ],
        key=lambda image_id: area_fraction[image_id],
    )

    ordered_ids = sorted(ids, key=lambda image_id: (scores[image_id], image_id))
    for quantile in (0.2, 0.4, 0.6, 0.8):
        candidate = ordered_ids[round((len(ordered_ids) - 1) * quantile)]
        if candidate not in assignments:
            assignments[candidate] = [f"additional_score_quantile_{quantile:.1f}"]
    for candidate in ordered_ids:
        if len(assignments) >= 12:
            break
        if candidate not in assignments:
            assignments[candidate] = ["additional_representative"]
    return assignments


def _write_failure_outputs(
    root: Path,
    checkpoint_info: dict[str, Any],
    scores: dict[str, float],
    predictions: dict[str, list[tuple[float, float, float, float]]],
    ground_truth: dict[str, list[tuple[float, float, float, float]]],
    cam_outputs: dict[str, dict[str, Any]],
) -> None:
    image_ids = sorted(scores)
    predicted_areas = [
        _box_area(box) for boxes in predictions.values() for box in boxes
    ]
    ground_truth_areas = [
        _box_area(box) for boxes in ground_truth.values() for box in boxes
    ]
    per_image_area_ratios = [
        sum(_box_area(box) for box in predictions[image_id])
        / sum(_box_area(box) for box in ground_truth[image_id])
        for image_id in image_ids
    ]
    counts = {
        count: sum(len(predictions[image_id]) == count for image_id in image_ids)
        for count in (0, 1)
    }
    counts["multiple"] = sum(
        len(predictions[image_id]) > 1 for image_id in image_ids
    )
    total_predicted_area = sum(predicted_areas)
    total_ground_truth_area = sum(ground_truth_areas)
    assignments = _select_error_examples(
        scores,
        predictions,
        ground_truth,
        {image_id: cam_outputs[image_id]["image_size"] for image_id in image_ids},
    )
    ERROR_ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    index_path = ERROR_ANALYSIS_DIR / "examples.csv"
    with index_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=(
                "image_id",
                "case_types",
                "iou",
                "predicted_box_count",
                "ground_truth_box_count",
                "predicted_area_fraction",
                "visualization",
            ),
        )
        writer.writeheader()
        for index, (image_id, categories) in enumerate(assignments.items(), start=1):
            score = scores[image_id]
            filename = f"{index:02d}_{image_id}_iou_{score:.4f}.png"
            relative_visualization = (
                ERROR_ANALYSIS_DIR / filename
            ).as_posix()
            _visualize_example(
                root,
                image_id,
                cam_outputs[image_id],
                ground_truth[image_id],
                predictions[image_id],
                score,
                root / relative_visualization,
            )
            height, width = cam_outputs[image_id]["image_size"]
            writer.writerow(
                {
                    "image_id": image_id,
                    "case_types": ";".join(categories),
                    "iou": score,
                    "predicted_box_count": len(predictions[image_id]),
                    "ground_truth_box_count": len(ground_truth[image_id]),
                    "predicted_area_fraction": (
                        sum(_box_area(box) for box in predictions[image_id])
                        / float(height * width)
                    ),
                    "visualization": relative_visualization,
                }
            )

    stats = {
        "dataset_split": "fixed test split, Pneumonia-positive images only",
        "num_images": len(image_ids),
        "threshold": 0.5,
        "baseline_result_file": BASELINE_PATH.as_posix(),
        "checkpoint_sha256": checkpoint_info["checkpoint_sha256"],
        "predicted_box_count_distribution": {
            "no_box": {
                "count": counts[0],
                "percentage": counts[0] / len(image_ids) * 100.0,
            },
            "one_box": {
                "count": counts[1],
                "percentage": counts[1] / len(image_ids) * 100.0,
            },
            "multiple_boxes": {
                "count": counts["multiple"],
                "percentage": counts["multiple"] / len(image_ids) * 100.0,
            },
        },
        "predicted_box_area_pixels_squared": {
            "definition": "Across all retained predicted boxes at the original DICOM resolution",
            "count": len(predicted_areas),
            "mean": float(np.mean(predicted_areas)) if predicted_areas else None,
            "median": float(np.median(predicted_areas)) if predicted_areas else None,
        },
        "ground_truth_box_area_pixels_squared": {
            "definition": "Across all positive RSNA ground-truth boxes at the original DICOM resolution",
            "count": len(ground_truth_areas),
            "mean": float(np.mean(ground_truth_areas)),
            "median": float(np.median(ground_truth_areas)),
        },
        "predicted_to_ground_truth_area_ratio": {
            "definition": "Per-image total predicted box area divided by total ground-truth box area; includes images with no predicted boxes",
            "mean": float(np.mean(per_image_area_ratios)),
            "median": float(np.median(per_image_area_ratios)),
            "ratio_of_total_areas": total_predicted_area / total_ground_truth_area,
        },
        "selected_example_count": len(assignments),
        "example_index": (ERROR_ANALYSIS_DIR / "examples.csv").as_posix(),
        "phase1_cam_results_modified": False,
        "failure_categories": {
            "irrelevant_anatomical_activation": (
                "The selected example's hotspot is centered below the annotated pulmonary opacity, in lower central/diaphragmatic anatomy."
            )
        },
    }
    (root / FAILURE_STATS_PATH).write_text(
        json.dumps(stats, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(f"Saved failure statistics: {root / FAILURE_STATS_PATH}")
    print(f"Saved {len(assignments)} examples and index: {index_path}")


def main() -> int:
    root = PROJECT_ROOT.resolve()
    baseline_path = root / BASELINE_PATH
    if not baseline_path.is_file():
        raise FileNotFoundError(f"Saved Phase 1 CAM results are required: {baseline_path}")
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))

    model, device, checkpoint_info = load_cnn_checkpoint(root)
    stats = load_normalization_stats(root / DEFAULT_STATS_PATH)
    if stats.source_split != "train" or stats.image_size != INPUT_SIZE:
        raise ValueError("CAM input must use the existing train-fitted 128x128 stats")
    positives = _read_test_positive_images(root)
    image_ids = {record["image_id"] for record in positives}
    ground_truth = _read_source_ground_truth(root, image_ids)
    if len(positives) != int(baseline["num_images"]):
        raise ValueError("Fixed test positive image count differs from saved baseline")

    print(
        f"Generating one CAM per existing test-positive image "
        f"({len(positives):,}); no model training will be performed.",
        flush=True,
    )
    cam_outputs = _generate_cams(
        root, model, device, stats, positives, batch_size=32
    )
    if set(cam_outputs) != set(ground_truth):
        raise AssertionError("CAM output IDs do not match fixed test annotations")

    threshold_rows: list[dict[str, float | int]] = []
    baseline_scores: dict[str, float] | None = None
    baseline_predictions: dict[
        str, list[tuple[float, float, float, float]]
    ] | None = None
    for threshold in THRESHOLDS:
        predictions = _make_predictions(
            cam_outputs,
            LocalizationConfig(threshold=threshold, batch_size=32),
        )
        metrics, _ = _baseline_metrics(predictions, ground_truth)
        scores = {
            image_id: image_iou(predictions[image_id], ground_truth[image_id])
            for image_id in sorted(ground_truth)
        }
        threshold_rows.append(
            {
                "threshold": threshold,
                **metrics,
                "num_images": len(scores),
                "num_images_with_no_predicted_boxes": sum(
                    not predictions[image_id] for image_id in predictions
                ),
            }
        )
        if threshold == 0.5:
            _verify_baseline(
                baseline,
                scores,
                {image_id: len(boxes) for image_id, boxes in predictions.items()},
            )
            baseline_scores = scores
            baseline_predictions = predictions
        print(
            f"threshold={threshold:.2f}: mean IoU={metrics['mean_iou']:.6f}, "
            f"median IoU={metrics['median_iou']:.6f}, "
            f"IoU>0={metrics['percentage_iou_gt_0']:.2f}%, "
            f"IoU>=0.5={metrics['percentage_iou_ge_0_5']:.2f}%",
            flush=True,
        )

    if baseline_scores is None or baseline_predictions is None:
        raise AssertionError("The required baseline threshold 0.5 was not evaluated")
    threshold_csv = root / THRESHOLD_CSV_PATH
    threshold_csv.parent.mkdir(parents=True, exist_ok=True)
    with threshold_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=threshold_rows[0].keys())
        writer.writeheader()
        writer.writerows(threshold_rows)
    _write_failure_outputs(
        root,
        checkpoint_info,
        baseline_scores,
        baseline_predictions,
        ground_truth,
        cam_outputs,
    )
    best = max(
        threshold_rows,
        key=lambda row: (
            float(row["mean_iou"]),
            float(row["median_iou"]),
            float(row["percentage_iou_ge_0_5"]),
            -float(row["threshold"]),
        ),
    )
    print(f"Saved threshold analysis: {threshold_csv}")
    print(f"Best threshold by mean IoU: {best['threshold']:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
