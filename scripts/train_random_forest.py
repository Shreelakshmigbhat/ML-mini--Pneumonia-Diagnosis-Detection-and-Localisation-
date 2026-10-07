"""Train and evaluate Random Forest on the project's fixed split."""

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
import sklearn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import PneumoniaDataset, SPLIT_METADATA
from src.data.preprocessing import DEFAULT_STATS_PATH, load_normalization_stats
from src.models.random_forest import RandomForestBaseline, RandomForestConfig

RESULTS_PATH = Path("results") / "random_forest.json"
MODEL_PATH = Path("results") / "random_forest_model.joblib"
CLASSIFICATION_RESULTS_PATH = Path("results") / "classification_results.csv"
CLASSIFICATION_COLUMNS = (
    "model",
    "train_accuracy",
    "validation_accuracy",
    "test_accuracy",
    "reference_test_accuracy",
    "difference_from_reference",
)
REFERENCE_TEST_ACCURACY = 0.83


def _load_features(
    dataset: PneumoniaDataset,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    features = np.empty((len(dataset), 128 * 128), dtype=np.float32)
    labels = np.empty(len(dataset), dtype=np.int8)
    image_ids: list[str] = []
    for index in range(len(dataset)):
        sample = dataset[index]
        image = sample["image"]
        if image.shape != (128, 128):
            raise ValueError(
                f"Unexpected image shape for {sample['image_id']}: {image.shape}"
            )
        features[index] = image.reshape(-1)
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


def _update_classification_results(
    root: Path, metrics: dict[str, float]
) -> None:
    result_path = root / CLASSIFICATION_RESULTS_PATH
    if not result_path.is_file():
        raise FileNotFoundError(
            f"Existing baseline comparison is missing: {result_path}"
        )
    with result_path.open("r", newline="", encoding="utf-8-sig") as csv_file:
        reader = csv.DictReader(csv_file)
        if reader.fieldnames != list(CLASSIFICATION_COLUMNS):
            raise ValueError(
                "Existing classification comparison has an unexpected schema"
            )
        rows = list(reader)

    rows = [row for row in rows if row["model"] != "Random Forest"]
    rows.append(
        {
            "model": "Random Forest",
            "train_accuracy": str(metrics["train_accuracy"]),
            "validation_accuracy": str(metrics["validation_accuracy"]),
            "test_accuracy": str(metrics["test_accuracy"]),
            "reference_test_accuracy": str(REFERENCE_TEST_ACCURACY),
            "difference_from_reference": str(
                metrics["test_accuracy"] - REFERENCE_TEST_ACCURACY
            ),
        }
    )
    with result_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CLASSIFICATION_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def run(config: RandomForestConfig) -> dict[str, Any]:
    root = PROJECT_ROOT
    split_path = root / SPLIT_METADATA
    split_digest = _split_hash(split_path)
    expected_results: dict[str, dict[str, Any]] = {}
    for name, filename in (
        ("Logistic Regression", "logistic_regression.json"),
        ("SVM", "svm.json"),
    ):
        result_path = root / "results" / filename
        if not result_path.is_file():
            raise FileNotFoundError(f"Existing {name} result is missing: {result_path}")
        previous_result = json.loads(result_path.read_text(encoding="utf-8"))
        previous_dataset = previous_result["dataset"]
        if previous_dataset["split_metadata_sha256"] != split_digest:
            raise ValueError(f"Fixed split differs from the recorded {name} run")
        expected_results[name] = previous_result

    logistic_dataset = expected_results["Logistic Regression"]["dataset"]
    svm_dataset = expected_results["SVM"]["dataset"]
    if logistic_dataset["normalization"] != svm_dataset["normalization"]:
        raise ValueError("Existing baselines used different normalization statistics")
    stats_path = root / DEFAULT_STATS_PATH
    stats = load_normalization_stats(stats_path)
    expected_normalization = logistic_dataset["normalization"]
    if (
        stats.mean != expected_normalization["mean"]
        or stats.std != expected_normalization["std"]
        or stats.source_split != "train"
        or stats.image_size != (128, 128)
    ):
        raise ValueError(
            "Shared preprocessing statistics differ from existing baselines"
        )
    if logistic_dataset["feature_count"] != 128 * 128:
        raise ValueError("Existing baseline feature shape is not 128x128 flattened")

    train_dataset = PneumoniaDataset(root, "train", stats, SPLIT_METADATA)
    validation_dataset = PneumoniaDataset(root, "validation", stats, SPLIT_METADATA)
    train_features, train_labels, train_ids = _load_features(train_dataset)
    validation_features, validation_labels, validation_ids = _load_features(
        validation_dataset
    )
    if set(train_ids) & set(validation_ids):
        raise ValueError("Train/validation image ID overlap found")

    model = RandomForestBaseline(config)
    print(
        "Fitting Random Forest on train only: "
        f"{len(train_labels):,} images, {train_features.shape[1]:,} features."
    )
    model.fit(train_features, train_labels)
    train_accuracy = model.score(train_features, train_labels)
    validation_accuracy = model.score(validation_features, validation_labels)
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

    result: dict[str, Any] = {
        "model": "random forest classifier",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": {
            "split_metadata": SPLIT_METADATA.as_posix(),
            "split_metadata_sha256": split_digest,
            "input_split_counts": {
                "train": len(train_ids),
                "validation": len(validation_ids),
                "test": len(test_ids),
            },
            "label_mapping": {
                "0": "Healthy/Normal",
                "1": "Pneumonia/Lung Opacity",
            },
            "feature_representation": "flattened normalized grayscale pixels",
            "feature_shape_per_image": [128, 128],
            "feature_count": 128 * 128,
            "normalization": {
                "method": "training-split global pixel mean and standard deviation",
                "stats_path": DEFAULT_STATS_PATH.as_posix(),
                "mean": stats.mean,
                "std": stats.std,
                "source_split": stats.source_split,
                "fit_image_count": expected_normalization["fit_image_count"],
            },
        },
        "configuration": {
            **model.configuration(),
            "model_selection": "No hyperparameter/model selection; validation used for evaluation only",
            "evaluation_protocol": {
                "fit_split": "train",
                "validation_split": "validation (evaluation only)",
                "final_evaluation_split": "test (after fitting)",
            },
            "reference_hyperparameters_specified": False,
            "configuration_note": (
                "The supplied reference description reports Random Forest "
                "accuracies but does not specify hyperparameters. We use 100 "
                "trees, Gini impurity, sqrt feature subsampling, maximum depth "
                "20, and minimum leaf size 2 as a deterministic, bounded-depth "
                "baseline for the high-dimensional flattened pixels. No "
                "validation-based parameter search was performed."
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
        "reference_result": {
            "train_accuracy": 0.8639,
            "train_accuracy_percent": 86.39,
            "test_accuracy": REFERENCE_TEST_ACCURACY,
            "test_accuracy_percent": REFERENCE_TEST_ACCURACY * 100,
        },
        "reference_cohort_match": False,
        "model_parameters_path": MODEL_PATH.as_posix(),
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
        },
    }

    results_directory = root / RESULTS_PATH.parent
    results_directory.mkdir(parents=True, exist_ok=True)
    model.save(root / MODEL_PATH)
    (root / RESULTS_PATH).write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    _update_classification_results(root, metrics)
    print(f"Train accuracy: {train_accuracy * 100:.2f}%")
    print(f"Validation accuracy: {validation_accuracy * 100:.2f}%")
    print(f"Test accuracy: {test_accuracy * 100:.2f}%")
    print(
        "Difference from reference test accuracy: "
        f"{(test_accuracy - REFERENCE_TEST_ACCURACY) * 100:+.2f} percentage points"
    )
    print(f"Saved metrics/configuration: {(root / RESULTS_PATH).relative_to(root)}")
    print(f"Saved model parameters: {(root / MODEL_PATH).relative_to(root)}")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Train and evaluate Random Forest on the fixed split."
    )
    parser.add_argument("--n-estimators", type=int, default=100)
    parser.add_argument("--max-depth", type=int, default=20)
    parser.add_argument("--max-features", choices=("sqrt", "log2"), default="sqrt")
    parser.add_argument("--min-samples-leaf", type=int, default=2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--n-jobs", type=int, default=-1)
    args = parser.parse_args(argv)
    try:
        run(
            RandomForestConfig(
                n_estimators=args.n_estimators,
                max_depth=args.max_depth,
                max_features=args.max_features,
                min_samples_leaf=args.min_samples_leaf,
                random_state=args.random_state,
                n_jobs=args.n_jobs,
            )
        )
    except (OSError, csv.Error, ValueError, KeyError, RuntimeError) as error:
        print(f"Random Forest run failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
