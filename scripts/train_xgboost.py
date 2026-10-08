"""Train the Stage 10 XGBoost extension on the fixed project split."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import sklearn
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from threadpoolctl import threadpool_limits
from xgboost import __version__ as xgboost_version

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import PneumoniaDataset, SPLIT_METADATA
from src.data.preprocessing import DEFAULT_STATS_PATH, load_normalization_stats
from src.models.cnn import Custom10LayerCNN
from src.models.logistic_regression import LogisticRegression
from src.models.svm import LinearSVM
from src.models.xgboost_model import (
    PCAConfig,
    XGBoostConfig,
    XGBoostPCAClassifier,
)

RESULTS_PATH = Path("results") / "xgboost_results.json"
COMPARISON_PATH = Path("results") / "model_comparison.csv"
MODEL_PATH = Path("results") / "xgboost_model.joblib"
CHECKPOINT_PATH = Path("models") / "cnn_best.pth"
FEATURE_DIMENSION = 128 * 128
LABELS = (0, 1)
COMPARISON_COLUMNS = (
    "model",
    "feature_representation",
    "test_accuracy",
    "test_f1",
    "test_precision",
    "test_recall",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_existing_results(root: Path, split_digest: str) -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    for name, filename in (
        ("Logistic Regression", "logistic_regression.json"),
        ("SVM", "svm.json"),
        ("Random Forest", "random_forest.json"),
    ):
        result_path = root / "results" / filename
        if not result_path.is_file():
            raise FileNotFoundError(f"Existing {name} result is missing: {result_path}")
        record = json.loads(result_path.read_text(encoding="utf-8"))
        if record["dataset"]["split_metadata_sha256"] != split_digest:
            raise ValueError(f"Existing {name} result uses a different split")
        results[name] = record
    return results


def _metrics(
    labels: np.ndarray,
    predictions: np.ndarray,
    probabilities: np.ndarray | None = None,
) -> dict[str, Any]:
    matrix = confusion_matrix(labels, predictions, labels=LABELS)
    result: dict[str, Any] = {
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "confusion_matrix": matrix.astype(int).tolist(),
        "class_counts": {
            "healthy": int(np.count_nonzero(labels == 0)),
            "pneumonia": int(np.count_nonzero(labels == 1)),
        },
    }
    if probabilities is not None and np.unique(labels).size == 2:
        result["roc_auc"] = float(roc_auc_score(labels, probabilities))
        false_positive_rate, true_positive_rate, _ = roc_curve(
            labels, probabilities
        )
        result["roc_curve"] = {
            "false_positive_rate": false_positive_rate.tolist(),
            "true_positive_rate": true_positive_rate.tolist(),
        }
    return result


def _load_raw_split(
    dataset: PneumoniaDataset,
    output_path: Path,
) -> tuple[np.memmap, np.ndarray, list[str]]:
    features = np.memmap(
        output_path,
        dtype=np.float32,
        mode="w+",
        shape=(len(dataset), FEATURE_DIMENSION),
    )
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
            print(
                f"Loaded {dataset.split}: {index + 1:,}/{len(dataset):,}",
                flush=True,
            )
    features.flush()
    return features, labels, image_ids


def _close_memmap(features: np.memmap) -> None:
    features.flush()
    mapping = getattr(features, "_mmap", None)
    if mapping is not None:
        mapping.close()


def _predict_linear_artifact(
    root: Path,
    filename: str,
    model_class: type[LogisticRegression] | type[LinearSVM],
    test_features: np.ndarray,
) -> np.ndarray:
    artifact_path = root / "results" / filename
    if not artifact_path.is_file():
        raise FileNotFoundError(f"Saved baseline parameters are missing: {artifact_path}")
    model = model_class()
    with np.load(artifact_path, allow_pickle=False) as saved:
        coefficients = saved["coefficients"].copy()
        intercept = float(saved["intercept"])
    model.coef_ = coefficients
    model.intercept_ = intercept
    return model.predict(test_features)


def _evaluate_existing_models(
    root: Path,
    existing_results: dict[str, dict[str, Any]],
    test_features: np.ndarray,
    test_labels: np.ndarray,
    checkpoint_path: Path,
) -> tuple[dict[str, dict[str, Any]], int]:
    models: dict[str, dict[str, Any]] = {}
    baseline_specs = (
        (
            "Logistic Regression",
            "logistic_regression_model.npz",
            LogisticRegression,
            "Flattened normalized pixels (16,384)",
        ),
        (
            "SVM",
            "svm_model.npz",
            LinearSVM,
            "Flattened normalized pixels (16,384)",
        ),
    )
    for name, filename, model_class, representation in baseline_specs:
        predictions = _predict_linear_artifact(
            root, filename, model_class, test_features
        )
        metrics = _metrics(test_labels, predictions)
        recorded_accuracy = float(existing_results[name]["metrics"]["test_accuracy"])
        if not np.isclose(metrics["accuracy"], recorded_accuracy, atol=1e-8):
            raise ValueError(
                f"Reused {name} artifact accuracy ({metrics['accuracy']}) does not "
                f"match its recorded result ({recorded_accuracy})"
            )
        models[name] = {"representation": representation, "metrics": metrics}

    forest_path = root / "results" / "random_forest_model.joblib"
    if not forest_path.is_file():
        raise FileNotFoundError(f"Saved Random Forest model is missing: {forest_path}")
    forest_payload = joblib.load(forest_path)
    if not isinstance(forest_payload, dict) or "estimator" not in forest_payload:
        raise ValueError("Saved Random Forest file has an unexpected format")
    forest_predictions = forest_payload["estimator"].predict(test_features)
    forest_metrics = _metrics(test_labels, forest_predictions)
    recorded_forest_accuracy = float(
        existing_results["Random Forest"]["metrics"]["test_accuracy"]
    )
    if not np.isclose(
        forest_metrics["accuracy"], recorded_forest_accuracy, atol=1e-8
    ):
        raise ValueError(
            "Reused Random Forest artifact does not match the saved test accuracy"
        )
    models["Random Forest"] = {
        "representation": "Flattened normalized pixels (16,384)",
        "metrics": forest_metrics,
    }

    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Saved CNN checkpoint is missing: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint["split_metadata_sha256"] != existing_results[
        "Logistic Regression"
    ]["dataset"]["split_metadata_sha256"]:
        raise ValueError("CNN checkpoint was created from a different split")
    cnn = Custom10LayerCNN()
    cnn.load_state_dict(checkpoint["state_dict"], strict=True)
    cnn.eval()
    torch.set_num_threads(min(4, torch.get_num_threads()))
    cnn_predictions: list[np.ndarray] = []
    with torch.inference_mode(), threadpool_limits(limits=4):
        for start in range(0, len(test_features), 64):
            batch = np.asarray(test_features[start : start + 64])
            images = torch.from_numpy(batch.reshape(-1, 1, 128, 128))
            logits = cnn(images)
            cnn_predictions.append(logits.argmax(dim=1).numpy().astype(np.int8))
    cnn_metrics = _metrics(test_labels, np.concatenate(cnn_predictions))
    models["CNN"] = {
        "representation": "Learned 10-layer CNN features (128x128 input)",
        "metrics": cnn_metrics,
        "checkpoint_epoch": int(checkpoint["epoch"]),
    }
    return models, int(checkpoint["epoch"])


def _write_comparison(
    root: Path, models: dict[str, dict[str, Any]]
) -> None:
    order = ("Logistic Regression", "SVM", "Random Forest", "CNN", "XGBoost")
    path = root / COMPARISON_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=COMPARISON_COLUMNS)
        writer.writeheader()
        for name in order:
            entry = models[name]
            metrics = entry["metrics"]
            writer.writerow(
                {
                    "model": name,
                    "feature_representation": entry["representation"],
                    "test_accuracy": f"{metrics['accuracy']:.10f}",
                    "test_f1": f"{metrics['f1']:.10f}",
                    "test_precision": f"{metrics['precision']:.10f}",
                    "test_recall": f"{metrics['recall']:.10f}",
                }
            )


def _balanced_indices(dataset: PneumoniaDataset, samples_per_class: int) -> np.ndarray:
    labels = np.asarray([int(row["label"]) for row in dataset.records])
    rng = np.random.default_rng(42 + samples_per_class)
    selected: list[int] = []
    for label in LABELS:
        available = np.flatnonzero(labels == label)
        if len(available) < samples_per_class:
            raise ValueError(f"Not enough class-{label} samples for smoke test")
        selected.extend(
            rng.choice(available, size=samples_per_class, replace=False).tolist()
        )
    return np.asarray(sorted(selected), dtype=np.int64)


def _load_subset(
    dataset: PneumoniaDataset, indices: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    features = np.empty((len(indices), FEATURE_DIMENSION), dtype=np.float32)
    labels = np.empty(len(indices), dtype=np.int8)
    for output_index, dataset_index in enumerate(indices):
        sample = dataset[int(dataset_index)]
        features[output_index] = sample["image"].reshape(-1)
        labels[output_index] = sample["label"]
    return features, labels


def run_smoke_test(root: Path = PROJECT_ROOT) -> dict[str, Any]:
    """Run PCA and XGBoost on tiny balanced subsets without saving artifacts."""
    root = root.resolve()
    stats = load_normalization_stats(root / DEFAULT_STATS_PATH)
    datasets = {
        split: PneumoniaDataset(root, split, stats, SPLIT_METADATA)
        for split in ("train", "validation", "test")
    }
    train_x, train_y = _load_subset(
        datasets["train"], _balanced_indices(datasets["train"], 250)
    )
    validation_x, validation_y = _load_subset(
        datasets["validation"], _balanced_indices(datasets["validation"], 50)
    )
    model = XGBoostPCAClassifier(
        PCAConfig(),
        XGBoostConfig(),
    )
    model.fit(train_x, train_y)
    predictions = model.predict(validation_x)
    if predictions.shape != validation_x.shape[:1]:
        raise AssertionError("XGBoost smoke test returned an unexpected prediction shape")
    result = {
        "train_samples": len(train_y),
        "validation_samples": len(validation_x),
        "smoke_validation_only": True,
        "pca_dimensions": [FEATURE_DIMENSION, model.pca.n_components_],
        "explained_variance_ratio": float(
            model.pca.explained_variance_ratio_.sum()
        ),
        "validation_accuracy": float(
            accuracy_score(validation_y, predictions)
        ),
        "validation_predictions": int(len(predictions)),
        "pca_fit_seconds": model.pca_fit_seconds_,
        "xgboost_fit_seconds": model.xgboost_fit_seconds_,
        "prediction_shape": list(predictions.shape),
    }
    print(f"Smoke test passed: {json.dumps(result, sort_keys=True)}")
    return result


def run_experiment(root: Path = PROJECT_ROOT) -> dict[str, Any]:
    root = root.resolve()
    split_path = root / SPLIT_METADATA
    split_digest = _sha256(split_path)
    existing_results = _load_existing_results(root, split_digest)
    reference_dataset = existing_results["Logistic Regression"]["dataset"]
    stats = load_normalization_stats(root / DEFAULT_STATS_PATH)
    recorded_stats = reference_dataset["normalization"]
    if (
        stats.mean != recorded_stats["mean"]
        or stats.std != recorded_stats["std"]
        or stats.source_split != "train"
        or stats.image_size != (128, 128)
    ):
        raise ValueError("Shared preprocessing differs from existing baselines")
    if reference_dataset["feature_count"] != FEATURE_DIMENSION:
        raise ValueError("Existing baseline does not use flattened 128x128 pixels")

    datasets = {
        split: PneumoniaDataset(root, split, stats, SPLIT_METADATA)
        for split in ("train", "validation", "test")
    }
    all_ids = {
        split: {record["image_id"] for record in dataset.records}
        for split, dataset in datasets.items()
    }
    if (
        all_ids["train"] & all_ids["validation"]
        or all_ids["train"] & all_ids["test"]
        or all_ids["validation"] & all_ids["test"]
    ):
        raise ValueError("The fixed split has image-ID overlap")
    split_counts = {split: len(dataset) for split, dataset in datasets.items()}
    if split_counts != reference_dataset["input_split_counts"]:
        raise ValueError("Current split counts differ from saved baseline results")

    models: dict[str, dict[str, Any]] = {}
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="pneumonia-xgboost-") as temporary:
        raw_features: dict[str, np.memmap] = {}
        labels: dict[str, np.ndarray] = {}
        ids: dict[str, list[str]] = {}
        try:
            image_load_started = time.perf_counter()
            for split in ("train", "validation", "test"):
                raw_features[split], labels[split], ids[split] = _load_raw_split(
                    datasets[split], Path(temporary) / f"{split}.f32"
                )
                print(
                    f"Finished {split} preprocessing in "
                    f"{time.perf_counter() - image_load_started:.1f}s.",
                    flush=True,
                )
                image_load_started = time.perf_counter()
            if (
                set(ids["train"]) & set(ids["validation"])
                or set(ids["train"]) & set(ids["test"])
                or set(ids["validation"]) & set(ids["test"])
            ):
                raise ValueError("Loaded image IDs overlap across split assignments")

            comparison_started = time.perf_counter()
            models, cnn_checkpoint_epoch = _evaluate_existing_models(
                root,
                existing_results,
                raw_features["test"],
                labels["test"],
                root / CHECKPOINT_PATH,
            )
            print(
                f"Reused-model prediction and comparison took "
                f"{time.perf_counter() - comparison_started:.1f}s.",
                flush=True,
            )

            pca_config = PCAConfig()
            xgboost_config = XGBoostConfig()
            model = XGBoostPCAClassifier(pca_config, xgboost_config)
            print(
                "Fitting randomized PCA on train only: "
                f"{len(labels['train']):,} x {FEATURE_DIMENSION:,} features.",
                flush=True,
            )
            xgb_started = time.perf_counter()
            model.fit(raw_features["train"], labels["train"])
            fit_seconds = time.perf_counter() - xgb_started
            print(
                f"PCA fit: {model.pca_fit_seconds_:.1f}s; "
                f"XGBoost fit: {model.xgboost_fit_seconds_:.1f}s.",
                flush=True,
            )

            print("Transforming validation and test with the fitted train PCA.")
            validation_pca = model.transform(raw_features["validation"])
            test_pca = model.transform(raw_features["test"])
            train_predictions = model.training_predictions_
            train_probabilities = model.training_probabilities_
            if train_predictions is None or train_probabilities is None:
                raise RuntimeError("Training predictions were not retained")
            train_metrics = _metrics(
                labels["train"], train_predictions, train_probabilities[:, 1]
            )
            validation_probabilities = model.predict_proba_transformed(validation_pca)[
                :, 1
            ]
            validation_predictions = model.predict_transformed(validation_pca)
            test_probabilities = model.predict_proba_transformed(test_pca)[:, 1]
            test_predictions = model.predict_transformed(test_pca)
            validation_metrics = _metrics(
                labels["validation"],
                validation_predictions,
                validation_probabilities,
            )
            test_metrics = _metrics(
                labels["test"], test_predictions, test_probabilities
            )

            models["XGBoost"] = {
                "representation": (
                    f"PCA ({model.pca.n_components_} components) "
                    "of flattened normalized pixels"
                ),
                "metrics": test_metrics,
            }
            result = {
                "model": "XGBoost extension (not part of the reference paper)",
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "random_seed": xgboost_config.random_state,
                "dataset": {
                    "split_metadata": SPLIT_METADATA.as_posix(),
                    "split_metadata_sha256": split_digest,
                    "label_mapping": {
                        "0": "Healthy/Normal",
                        "1": "Pneumonia/Lung Opacity",
                    },
                    "normalization": {
                        "method": "shared train-split global mean and standard deviation",
                        "stats_path": DEFAULT_STATS_PATH.as_posix(),
                        "mean": stats.mean,
                        "std": stats.std,
                        "source_split": stats.source_split,
                        "fit_image_count": recorded_stats["fit_image_count"],
                    },
                },
                "sample_counts": {
                    "train": split_counts["train"],
                    "validation": split_counts["validation"],
                    "test": split_counts["test"],
                },
                "original_feature_dimension": FEATURE_DIMENSION,
                "pca_feature_dimension": int(model.pca.n_components_),
                "pca_configuration": {
                    **model.configuration()["pca"],
                    "fit_sample_count": split_counts["train"],
                    "explained_variance_ratio": model.pca.explained_variance_ratio_.tolist(),
                    "total_explained_variance_ratio": float(
                        model.pca.explained_variance_ratio_.sum()
                    ),
                },
                "xgboost_parameters": model.configuration()["xgboost"],
                "train_metrics": train_metrics,
                "validation_metrics": validation_metrics,
                "test_metrics": test_metrics,
                "training_time_seconds": model.xgboost_fit_seconds_,
                "pca_fit_time_seconds": model.pca_fit_seconds_,
                "xgboost_fit_time_seconds": model.xgboost_fit_seconds_,
                "pca_plus_xgboost_fit_time_seconds": fit_seconds,
                "full_run_elapsed_seconds": time.perf_counter() - started,
                "comparison_models": models,
                "cnn_comparison_checkpoint_epoch": cnn_checkpoint_epoch,
                "evaluation_notes": [
                    "PCA was fitted using the training feature matrix only.",
                    "Validation and test matrices were transformed using that fitted PCA.",
                    "No previous model was retrained; saved LR/SVM/RF artifacts were reused.",
                    "The existing CNN checkpoint was evaluated by inference only.",
                    "The test split was used only for final evaluation and comparison.",
                ],
                "software": {
                    "python": platform.python_version(),
                    "numpy": np.__version__,
                    "scikit_learn": sklearn.__version__,
                    "xgboost": xgboost_version,
                },
                "model_path": MODEL_PATH.as_posix(),
            }

            results_dir = root / RESULTS_PATH.parent
            results_dir.mkdir(parents=True, exist_ok=True)
            joblib.dump(model, root / MODEL_PATH, compress=3)
            (root / RESULTS_PATH).write_text(
                json.dumps(result, indent=2, allow_nan=False) + "\n",
                encoding="utf-8",
            )
            _write_comparison(root, models)
            for split in ("train", "validation", "test"):
                print(
                    f"{split.title()} accuracy: "
                    f"{result[f'{split}_metrics']['accuracy']:.4%}"
                )
            print(
                f"Test F1 / precision / recall: {test_metrics['f1']:.4f} / "
                f"{test_metrics['precision']:.4f} / {test_metrics['recall']:.4f}"
            )
            print(
                "PCA components / explained variance: "
                f"{model.pca.n_components_} / "
                f"{result['pca_configuration']['total_explained_variance_ratio']:.4%}"
            )
            print(f"XGBoost fit time: {model.xgboost_fit_seconds_:.2f}s")
            print(f"Saved results: {root / RESULTS_PATH}")
            print(f"Saved comparison: {root / COMPARISON_PATH}")
            print(f"Saved model and PCA: {root / MODEL_PATH}")
            return result
        finally:
            for features in raw_features.values():
                _close_memmap(features)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the Stage 10 XGBoost extension on the fixed split."
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Fit PCA and XGBoost on small balanced subsets without saving results.",
    )
    args = parser.parse_args(argv)
    try:
        if args.smoke_test:
            run_smoke_test()
        else:
            run_experiment()
    except (OSError, csv.Error, ValueError, KeyError, RuntimeError) as error:
        print(f"XGBoost run failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
