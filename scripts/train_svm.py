"""Train and evaluate a linear SVM on the fixed Stage 2 split."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import PneumoniaDataset, SPLIT_METADATA
from src.data.preprocessing import (
    DEFAULT_STATS_PATH,
    load_normalization_stats,
)
from src.models.svm import LinearSVM, SVMConfig

RESULTS_PATH = Path("results") / "svm.json"
MODEL_PATH = Path("results") / "svm_model.npz"
CLASSIFICATION_RESULTS_PATH = Path("results") / "classification_results.csv"
LOGISTIC_RESULTS_PATH = Path("results") / "logistic_regression.json"
CLASSIFICATION_COLUMNS = (
    "model",
    "train_accuracy",
    "validation_accuracy",
    "test_accuracy",
    "reference_test_accuracy",
    "difference_from_reference",
)


def _load_features(
    dataset: PneumoniaDataset,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    features = np.empty((len(dataset), 128 * 128), dtype=np.float32)
    labels = np.empty(len(dataset), dtype=np.int8)
    image_ids: list[str] = []
    for index in range(len(dataset)):
        sample = dataset[index]
        if sample["image"].shape != (128, 128):
            raise ValueError(
                f"Unexpected image shape for {sample['image_id']}: "
                f"{sample['image'].shape}"
            )
        features[index] = sample["image"].reshape(-1)
        labels[index] = sample["label"]
        image_ids.append(sample["image_id"])
        if (index + 1) % 1000 == 0 or index + 1 == len(dataset):
            print(f"Loaded {dataset.split}: {index + 1:,}/{len(dataset):,}")
    return features, labels, image_ids


def _split_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as split_file:
        for chunk in iter(lambda: split_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _class_counts(labels: np.ndarray) -> dict[str, int]:
    return {
        "healthy": int(np.count_nonzero(labels == 0)),
        "pneumonia": int(np.count_nonzero(labels == 1)),
    }


def _write_classification_results(
    root: Path,
    *,
    svm_metrics: dict[str, float],
    svm_reference_test: float,
) -> None:
    rows: list[dict[str, str]] = []
    existing_path = root / CLASSIFICATION_RESULTS_PATH
    if existing_path.exists():
        with existing_path.open("r", newline="", encoding="utf-8-sig") as result_file:
            reader = csv.DictReader(result_file)
            if reader.fieldnames != list(CLASSIFICATION_COLUMNS):
                raise ValueError(
                    "Existing classification_results.csv has an unexpected schema"
                )
            rows = list(reader)
    else:
        logistic_path = root / LOGISTIC_RESULTS_PATH
        if not logistic_path.is_file():
            raise FileNotFoundError(
                f"Required Logistic Regression results are missing: {logistic_path}"
            )
        logistic = json.loads(logistic_path.read_text(encoding="utf-8"))
        logistic_metrics = logistic["metrics"]
        logistic_reference = float(
            logistic["reference_result_percent"]["test"]
        ) / 100.0
        rows.append(
            {
                "model": "Logistic Regression",
                "train_accuracy": str(logistic_metrics["train_accuracy"]),
                "validation_accuracy": str(logistic_metrics["validation_accuracy"]),
                "test_accuracy": str(logistic_metrics["test_accuracy"]),
                "reference_test_accuracy": str(logistic_reference),
                "difference_from_reference": str(
                    float(logistic_metrics["test_accuracy"]) - logistic_reference
                ),
            }
        )

    rows = [row for row in rows if row["model"] != "SVM"]
    rows.append(
        {
            "model": "SVM",
            "train_accuracy": str(svm_metrics["train_accuracy"]),
            "validation_accuracy": str(svm_metrics["validation_accuracy"]),
            "test_accuracy": str(svm_metrics["test_accuracy"]),
            "reference_test_accuracy": str(svm_reference_test),
            "difference_from_reference": str(
                svm_metrics["test_accuracy"] - svm_reference_test
            ),
        }
    )
    with existing_path.open("w", newline="", encoding="utf-8") as result_file:
        writer = csv.DictWriter(result_file, fieldnames=CLASSIFICATION_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def run(config: SVMConfig) -> dict[str, Any]:
    root = PROJECT_ROOT
    split_path = root / SPLIT_METADATA
    split_digest = _split_hash(split_path)
    logistic_path = root / LOGISTIC_RESULTS_PATH
    if not logistic_path.is_file():
        raise FileNotFoundError(
            f"Logistic Regression results are required to verify shared inputs: "
            f"{logistic_path}"
        )
    logistic = json.loads(logistic_path.read_text(encoding="utf-8"))
    logistic_dataset = logistic["dataset"]
    if logistic_dataset["split_metadata_sha256"] != split_digest:
        raise ValueError("Fixed split differs from the Logistic Regression run")

    stats_path = root / DEFAULT_STATS_PATH
    if not stats_path.is_file():
        raise FileNotFoundError(
            f"Shared preprocessing statistics are missing: {stats_path}"
        )
    stats = load_normalization_stats(stats_path)
    expected_stats = logistic_dataset["normalization"]
    if (
        stats.mean != expected_stats["mean"]
        or stats.std != expected_stats["std"]
        or stats.source_split != "train"
        or stats.image_size != (128, 128)
    ):
        raise ValueError(
            "Preprocessing statistics differ from the Logistic Regression run"
        )

    train_dataset = PneumoniaDataset(root, "train", stats, SPLIT_METADATA)
    validation_dataset = PneumoniaDataset(
        root, "validation", stats, SPLIT_METADATA
    )
    train_features, train_labels, train_ids = _load_features(train_dataset)
    validation_features, validation_labels, validation_ids = _load_features(
        validation_dataset
    )
    if set(train_ids) & set(validation_ids):
        raise ValueError("Train/validation image ID leakage found in fixed split")

    model = LinearSVM(config)
    model.fit(
        train_features,
        train_labels,
        validation_features,
        validation_labels,
    )
    train_accuracy = model.score(train_features, train_labels)
    validation_accuracy = model.best_validation_accuracy_
    if validation_accuracy is None:
        raise RuntimeError("SVM did not retain a validation score")
    del train_features
    del validation_features

    test_dataset = PneumoniaDataset(root, "test", stats, SPLIT_METADATA)
    test_features, test_labels, test_ids = _load_features(test_dataset)
    if set(train_ids) & set(test_ids) or set(validation_ids) & set(test_ids):
        raise ValueError("Test image ID overlaps another split")
    test_accuracy = model.score(test_features, test_labels)

    metrics = {
        "train_accuracy": train_accuracy,
        "validation_accuracy": validation_accuracy,
        "test_accuracy": test_accuracy,
    }
    reference = {"train": 0.7417, "test": 0.5818}
    result = {
        "model": "linear support vector machine",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": {
            "split_metadata": SPLIT_METADATA.as_posix(),
            "split_metadata_sha256": split_digest,
            "input_split_counts": {
                "train": len(train_ids),
                "validation": len(validation_ids),
                "test": len(test_ids),
            },
            "label_mapping": {"0": "Healthy/Normal", "1": "Pneumonia/Lung Opacity"},
            "feature_representation": "flattened normalized grayscale pixels",
            "feature_shape_per_image": [128, 128],
            "feature_count": 128 * 128,
            "normalization": {
                "method": "training-split global pixel mean and standard deviation",
                "stats_path": DEFAULT_STATS_PATH.as_posix(),
                "mean": stats.mean,
                "std": stats.std,
                "source_split": stats.source_split,
                "fit_image_count": expected_stats["fit_image_count"],
            },
        },
        "configuration": {
            **model.configuration(),
            "loss": "squared hinge",
            "objective": (
                "0.5 * ||weights||^2 + C * mean(max(0, 1 - y_signed * "
                "decision_function)^2)"
            ),
            "optimizer": "NumPy L-BFGS with Armijo backtracking",
            "fit_intercept": True,
            "class_weight": None,
            "model_selection": "best validation accuracy at configured checkpoints",
            "selection_iteration": model.best_iteration_,
            "evaluation_protocol": {
                "fit_split": "train",
                "model_selection_split": "validation",
                "final_evaluation_split": (
                    "test (evaluated once after model selection)"
                ),
            },
            "reference_hyperparameters_specified": False,
            "configuration_note": (
                "The supplied reference description does not specify SVM kernel "
                "or hyperparameters. A linear kernel and C=1.0 were selected "
                "for the flattened-pixel baseline. The linear squared-hinge "
                "primal objective and optimization settings are recorded here."
            ),
        },
        "metrics": {
            **metrics,
            "train_accuracy_percent": train_accuracy * 100,
            "validation_accuracy_percent": validation_accuracy * 100,
            "test_accuracy_percent": test_accuracy * 100,
        },
        "counts": {
            "train": {"total": len(train_labels), **_class_counts(train_labels)},
            "validation": {
                "total": len(validation_labels),
                **_class_counts(validation_labels),
            },
            "test": {"total": len(test_labels), **_class_counts(test_labels)},
        },
        "validation_history": model.validation_history_,
        "reference_result": {
            "train_accuracy": reference["train"],
            "train_accuracy_percent": reference["train"] * 100,
            "test_accuracy": reference["test"],
            "test_accuracy_percent": reference["test"] * 100,
        },
        "reference_cohort_match": False,
        "model_parameters_path": MODEL_PATH.as_posix(),
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
        },
    }

    results_dir = root / RESULTS_PATH.parent
    results_dir.mkdir(parents=True, exist_ok=True)
    model.save(str(root / MODEL_PATH))
    (root / RESULTS_PATH).write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    _write_classification_results(
        root,
        svm_metrics=metrics,
        svm_reference_test=reference["test"],
    )
    print(f"Train accuracy: {train_accuracy * 100:.2f}%")
    print(f"Validation accuracy: {validation_accuracy * 100:.2f}%")
    print(f"Test accuracy: {test_accuracy * 100:.2f}%")
    print(f"Validation-selected iteration: {model.best_iteration_}")
    print(f"Saved config and metrics: {(root / RESULTS_PATH).relative_to(root)}")
    print(f"Saved parameters: {(root / MODEL_PATH).relative_to(root)}")
    print(
        "Updated comparison table: "
        f"{(root / CLASSIFICATION_RESULTS_PATH).relative_to(root)}"
    )
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Train and evaluate fixed-split linear SVM."
    )
    parser.add_argument("--c", type=float, default=1.0)
    parser.add_argument("--max-iterations", type=int, default=100)
    parser.add_argument("--tolerance", type=float, default=1e-5)
    parser.add_argument("--validation-interval", type=int, default=5)
    parser.add_argument("--line-search-max-steps", type=int, default=30)
    parser.add_argument("--history-size", type=int, default=10)
    args = parser.parse_args(argv)
    try:
        config = SVMConfig(
            c=args.c,
            max_iterations=args.max_iterations,
            tolerance=args.tolerance,
            validation_interval=args.validation_interval,
            line_search_max_steps=args.line_search_max_steps,
            history_size=args.history_size,
        )
        run(config)
    except (OSError, csv.Error, ValueError, KeyError, RuntimeError) as error:
        print(f"SVM run failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
