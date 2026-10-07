"""Train and evaluate flattened-pixel logistic regression on the fixed split."""

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
    NormalizationStats,
    fit_training_stats_from_dataset,
    load_normalization_stats,
)
from src.models.logistic_regression import (
    LogisticRegression,
    LogisticRegressionConfig,
)

RESULTS_PATH = Path("results") / "logistic_regression.json"
MODEL_PATH = Path("results") / "logistic_regression_model.npz"


def _placeholder_stats() -> NormalizationStats:
    return NormalizationStats(mean=0.0, std=1.0, pixel_scale=1.0)


def _dataset_matrix(
    dataset: PneumoniaDataset,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    feature_count = 128 * 128
    features = np.empty((len(dataset), feature_count), dtype=np.float32)
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


def run(config: LogisticRegressionConfig) -> dict[str, Any]:
    root = PROJECT_ROOT
    split_path = root / SPLIT_METADATA
    stats_path = root / DEFAULT_STATS_PATH

    train_dataset = PneumoniaDataset(
        root, "train", _placeholder_stats(), SPLIT_METADATA
    )
    if stats_path.is_file():
        stats = load_normalization_stats(stats_path)
        print(f"Loaded train-only normalization stats: {stats_path}")
    else:
        print(
            f"Computing normalization statistics from "
            f"{len(train_dataset):,} train images."
        )
        stats = fit_training_stats_from_dataset(train_dataset, stats_path)
        print(f"Saved train-only normalization stats: {stats_path}")

    train_dataset = PneumoniaDataset(root, "train", stats, SPLIT_METADATA)
    validation_dataset = PneumoniaDataset(
        root, "validation", stats, SPLIT_METADATA
    )
    train_features, train_labels, train_ids = _dataset_matrix(train_dataset)
    validation_features, validation_labels, validation_ids = _dataset_matrix(
        validation_dataset
    )

    if set(train_ids) & set(validation_ids):
        raise ValueError("Train/validation image ID leakage found in fixed split")

    model = LogisticRegression(config)
    model.fit(
        train_features,
        train_labels,
        validation_features,
        validation_labels,
    )

    train_accuracy = model.score(train_features, train_labels)
    validation_accuracy = model.best_validation_accuracy_
    if validation_accuracy is None:
        raise RuntimeError("Model did not retain a validation score")

    del train_features
    del validation_features

    test_dataset = PneumoniaDataset(root, "test", stats, SPLIT_METADATA)
    test_features, test_labels, test_ids = _dataset_matrix(test_dataset)
    if (set(train_ids) & set(test_ids)) or (set(validation_ids) & set(test_ids)):
        raise ValueError("Test image ID overlaps another split")
    test_accuracy = model.score(test_features, test_labels)

    results_dir = root / RESULTS_PATH.parent
    results_dir.mkdir(parents=True, exist_ok=True)
    model_path = root / MODEL_PATH
    model.save(str(model_path))
    payload = {
        "model": "binary logistic regression",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": {
            "split_metadata": SPLIT_METADATA.as_posix(),
            "split_metadata_sha256": _split_hash(split_path),
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
                "fit_image_count": len(train_ids),
            },
        },
        "configuration": {
            **model.configuration(),
            "optimizer": "NumPy L-BFGS with Armijo backtracking",
            "model_selection": "best validation accuracy at configured checkpoints",
            "selection_iteration": model.best_iteration_,
            "evaluation_protocol": {
                "fit_split": "train",
                "model_selection_split": "validation",
                "final_evaluation_split": "test (evaluated once after model selection)",
            },
        },
        "metrics": {
            "train_accuracy": train_accuracy,
            "train_accuracy_percent": train_accuracy * 100,
            "validation_accuracy": validation_accuracy,
            "validation_accuracy_percent": validation_accuracy * 100,
            "test_accuracy": test_accuracy,
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
        "reference_result_percent": {"train": 75.86, "test": 73.02},
        "reference_cohort_match": False,
        "computational_modification": {
            "required": True,
            "details": (
                "scikit-learn LogisticRegression could not be imported because "
                "Windows Application Control blocked a SciPy DLL. The same "
                "L2-regularized binary logistic objective was optimized with a "
                "deterministic NumPy L-BFGS implementation instead; no features, "
                "splits, or evaluation roles were changed."
            ),
        },
        "model_parameters_path": MODEL_PATH.as_posix(),
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
        },
    }
    (root / RESULTS_PATH).write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        f"Train accuracy: {train_accuracy * 100:.2f}% "
        f"({len(train_labels):,} images)"
    )
    print(
        f"Validation accuracy: {validation_accuracy * 100:.2f}% "
        f"({len(validation_labels):,} images)"
    )
    print(
        f"Test accuracy: {test_accuracy * 100:.2f}% "
        f"({len(test_labels):,} images)"
    )
    print(f"Selected iteration: {model.best_iteration_}")
    print(f"Saved metrics/config: {(root / RESULTS_PATH).relative_to(root)}")
    print(f"Saved learned parameters: {model_path.relative_to(root)}")
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Train and evaluate fixed-split logistic regression."
    )
    parser.add_argument("--l2-strength", type=float, default=1e-4)
    parser.add_argument("--max-iterations", type=int, default=100)
    parser.add_argument("--tolerance", type=float, default=1e-5)
    parser.add_argument("--validation-interval", type=int, default=5)
    parser.add_argument("--line-search-max-steps", type=int, default=30)
    parser.add_argument("--history-size", type=int, default=10)
    args = parser.parse_args(argv)
    try:
        config = LogisticRegressionConfig(
            l2_strength=args.l2_strength,
            max_iterations=args.max_iterations,
            tolerance=args.tolerance,
            validation_interval=args.validation_interval,
            line_search_max_steps=args.line_search_max_steps,
            history_size=args.history_size,
        )
        run(config)
    except (OSError, csv.Error, ValueError, KeyError, RuntimeError) as error:
        print(f"Logistic regression run failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
