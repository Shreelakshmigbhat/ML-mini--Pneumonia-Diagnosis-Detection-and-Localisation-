"""Build a Stage 14 ablation summary from verified Stage 9/13 test outputs."""

from __future__ import annotations

import csv
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASELINE_PATH = Path("results") / "cam_results.json"
IMPROVED_PATH = Path("results") / "phase2_improved_cam_results.json"
CSV_PATH = Path("results") / "phase2_ablation.csv"
SUMMARY_PATH = Path("results") / "phase2_ablation_summary.json"
EXPECTED_THRESHOLD = 0.5
EXPECTED_IMAGE_COUNT = 601
EXPECTED_CHECKPOINT_SHA256 = (
    "75ae304214bbea8d9d8ec3857904419c4215d564ab6e50fc84f5ccd2b4c8833f"
)


def _load_json(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(f"Required localization result is missing: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def _require_close(
    actual: object,
    expected: object,
    *,
    field: str,
    tolerance: float = 1e-12,
) -> None:
    if abs(float(actual) - float(expected)) > tolerance:
        raise ValueError(f"Result mismatch for {field}: {actual} != {expected}")


def build_ablation(project_root: Path = PROJECT_ROOT) -> dict[str, object]:
    root = project_root.resolve()
    baseline = _load_json(root / BASELINE_PATH)
    improved = _load_json(root / IMPROVED_PATH)

    if baseline.get("threshold") != EXPECTED_THRESHOLD:
        raise ValueError("Expected the official Phase 1 CAM threshold of 0.50")
    if int(baseline.get("num_images", -1)) != EXPECTED_IMAGE_COUNT:
        raise ValueError("Phase 1 baseline does not cover the expected 601 images")
    if int(improved.get("test_sample_count", -1)) != EXPECTED_IMAGE_COUNT:
        raise ValueError("Phase 2 result does not cover the same 601 images")
    if improved.get("phase1_results_modified") is not False:
        raise ValueError("Stage 13 result does not attest Phase 1 outputs were preserved")
    if improved.get("ground_truth_used_for_generation_or_processing") is not False:
        raise ValueError("Stage 13 result indicates ground-truth processing leakage")
    baseline_checkpoint = baseline.get("checkpoint", {})
    improved_checkpoint = improved.get("checkpoint", {})
    if not isinstance(baseline_checkpoint, dict) or not isinstance(
        improved_checkpoint, dict
    ):
        raise ValueError("Missing checkpoint metadata in saved results")
    baseline_sha = baseline_checkpoint.get("checkpoint_sha256")
    improved_sha = improved_checkpoint.get("checkpoint_sha256")
    if baseline_sha != improved_sha or baseline_sha != EXPECTED_CHECKPOINT_SHA256:
        raise ValueError("Phase 1 and Phase 2 results do not use the same CNN checkpoint")
    if improved.get("split") != "existing fixed test split; 601 Pneumonia-positive images":
        raise ValueError("Phase 2 result does not identify the expected fixed test split")

    baseline_metrics = {
        "mean_iou": float(baseline["mean_iou"]),
        "median_iou": float(baseline["median_iou"]),
        "iou_gt_0": float(baseline["percentage_iou_gt_0"]),
        "iou_ge_0_5": float(baseline["percentage_iou_ge_0_5"]),
    }
    stage13_baseline_metrics = improved.get("baseline_metrics")
    if not isinstance(stage13_baseline_metrics, dict):
        raise ValueError("Stage 13 result is missing its baseline metrics")
    for metric, source_field in (
        ("mean_iou", "mean_iou"),
        ("median_iou", "median_iou"),
        ("iou_gt_0", "percentage_iou_gt_0"),
        ("iou_ge_0_5", "percentage_iou_ge_0_5"),
    ):
        _require_close(
            baseline_metrics[metric],
            stage13_baseline_metrics[source_field],
            field=f"Phase 1/Stage 13 baseline {metric}",
        )

    parameters = improved.get("parameters")
    if not isinstance(parameters, dict):
        raise ValueError("Stage 13 result is missing method parameters")
    if (
        float(parameters.get("threshold", -1)) != EXPECTED_THRESHOLD
        or parameters.get("apply_two_sigma_filter") is not True
        or parameters.get("connectivity") != 8
        or parameters.get("input_size") != [128, 128]
        or parameters.get("cam_output_size") != [128, 128]
    ):
        raise ValueError("Stage 13 post-processing differs from the expected control")
    if improved.get("evaluation_strategy") != baseline.get("iou_matching"):
        raise ValueError("Phase 1 and Phase 2 IoU matching strategies differ")

    improved_metrics = {
        "mean_iou": float(improved["mean_iou"]),
        "median_iou": float(improved["median_iou"]),
        "iou_gt_0": float(improved["percentage_iou_gt_0"]),
        "iou_ge_0_5": float(improved["percentage_iou_ge_0_5"]),
    }
    absolute_change = improved_metrics["mean_iou"] - baseline_metrics["mean_iou"]
    relative_change_percent = absolute_change / baseline_metrics["mean_iou"] * 100.0
    configurations = [
        {
            "name": "Phase 1 Baseline",
            "map_generation": "Classifier-weighted ReLU CAM",
            "changed_component": "None; official baseline",
            "post_processing": (
                "Threshold 0.50 inclusive; 8-connected DFS; enabled two-sigma "
                "cluster-size filter; original-DICOM box scaling"
            ),
            "metrics": baseline_metrics,
            "source": BASELINE_PATH.as_posix(),
            "interpretation": "Official Phase 1 control; saved result reused.",
        },
        {
            "name": "Ablation: Grad-CAM++ removed",
            "map_generation": "Classifier-weighted ReLU CAM",
            "changed_component": (
                "Grad-CAM++ map generation removed; all Stage 13 post-processing "
                "and evaluation settings retained"
            ),
            "post_processing": (
                "Threshold 0.50 inclusive; 8-connected DFS; enabled two-sigma "
                "cluster-size filter; original-DICOM box scaling"
            ),
            "metrics": baseline_metrics,
            "source": BASELINE_PATH.as_posix(),
            "interpretation": (
                "Algorithmically identical to Phase 1 Baseline because map "
                "generation is Stage 13's only changed component."
            ),
        },
        {
            "name": "Phase 2 Full Method",
            "map_generation": "Grad-CAM++ class-1 map",
            "changed_component": "Grad-CAM++ map generation enabled",
            "post_processing": (
                "Threshold 0.50 inclusive; 8-connected DFS; enabled two-sigma "
                "cluster-size filter; original-DICOM box scaling"
            ),
            "metrics": improved_metrics,
            "source": IMPROVED_PATH.as_posix(),
            "interpretation": "Stage 13 full-method test result reused.",
        },
    ]
    best = max(configurations, key=lambda item: item["metrics"]["mean_iou"])
    summary: dict[str, object] = {
        "stage": 14,
        "evaluation": {
            "test_sample_count": EXPECTED_IMAGE_COUNT,
            "split": "existing fixed test split; Pneumonia-positive images",
            "checkpoint_sha256": baseline_sha,
            "preprocessing": "Existing Stage 8 train-fitted statistics; 128x128",
            "iou_matching": baseline.get("iou_matching"),
            "ground_truth_used_for_prediction_or_configuration": False,
            "new_localization_runs_performed": 0,
            "reuse_note": (
                "The baseline is the saved Stage 9 result. Removing the sole "
                "Stage 13 change (Grad-CAM++ map generation) exactly restores "
                "the baseline algorithm, so the ablation has identical metrics. "
                "The Stage 13 full-method result was also reused; no duplicate "
                "601-image evaluation was run."
            ),
        },
        "configurations": configurations,
        "baseline_configuration": configurations[0]["name"],
        "best_configuration": best["name"],
        "improvement_over_baseline": {
            "mean_iou_absolute": absolute_change,
            "mean_iou_relative_percent": relative_change_percent,
            "median_iou_absolute": (
                improved_metrics["median_iou"] - baseline_metrics["median_iou"]
            ),
            "iou_gt_0_percentage_points": (
                improved_metrics["iou_gt_0"] - baseline_metrics["iou_gt_0"]
            ),
            "iou_ge_0_5_percentage_points": (
                improved_metrics["iou_ge_0_5"] - baseline_metrics["iou_ge_0_5"]
            ),
        },
        "interpretation": {
            "primary_component": "Grad-CAM++ map generation",
            "removing_primary_component_reduces_mean_iou": absolute_change > 0.0,
            "full_method_outperforms_baseline_mean_iou": absolute_change > 0.0,
            "improvement_consistent_across_all_metrics": (
                improved_metrics["mean_iou"] > baseline_metrics["mean_iou"]
                and improved_metrics["median_iou"] > baseline_metrics["median_iou"]
                and improved_metrics["iou_gt_0"] > baseline_metrics["iou_gt_0"]
                and improved_metrics["iou_ge_0_5"] > baseline_metrics["iou_ge_0_5"]
            ),
            "conclusion": (
                "The controlled results associate the mean, median, and any-overlap "
                "gains with Grad-CAM++ map generation; the IoU >= 0.5 rate is "
                "unchanged, so the improvement is not consistent across all metrics."
            ),
        },
    }

    csv_path = root / CSV_PATH
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        fieldnames = (
            "Method",
            "Mean IoU",
            "Median IoU",
            "IoU > 0",
            "IoU >= 0.5",
        )
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for configuration in configurations:
            metrics = configuration["metrics"]
            writer.writerow(
                {
                    "Method": configuration["name"],
                    "Mean IoU": metrics["mean_iou"],
                    "Median IoU": metrics["median_iou"],
                    "IoU > 0": metrics["iou_gt_0"],
                    "IoU >= 0.5": metrics["iou_ge_0_5"],
                }
            )
    (root / SUMMARY_PATH).write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> int:
    summary = build_ablation()
    print(f"Best configuration by mean IoU: {summary['best_configuration']}")
    print(f"Saved: {PROJECT_ROOT / CSV_PATH}")
    print(f"Saved: {PROJECT_ROOT / SUMMARY_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
