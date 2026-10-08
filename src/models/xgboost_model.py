"""Incremental-PCA plus XGBoost binary classifier for flattened X-rays."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from time import perf_counter
from typing import Any

import numpy as np
from sklearn.decomposition import PCA
from threadpoolctl import threadpool_limits
from xgboost import XGBClassifier


@dataclass(frozen=True)
class XGBoostConfig:
    """Bounded deterministic XGBoost configuration for this extension."""

    n_estimators: int = 100
    max_depth: int = 4
    learning_rate: float = 0.05
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    random_state: int = 42
    n_jobs: int = 4

    def __post_init__(self) -> None:
        if self.n_estimators <= 0 or self.max_depth <= 0:
            raise ValueError("n_estimators and max_depth must be positive")
        if not 0.0 < self.learning_rate <= 1.0:
            raise ValueError("learning_rate must be in (0, 1]")
        if not 0.0 < self.subsample <= 1.0:
            raise ValueError("subsample must be in (0, 1]")
        if not 0.0 < self.colsample_bytree <= 1.0:
            raise ValueError("colsample_bytree must be in (0, 1]")
        if self.n_jobs == 0:
            raise ValueError("n_jobs cannot be zero")


@dataclass(frozen=True)
class PCAConfig:
    """Randomized train-only PCA settings."""

    n_components: int = 256
    iterated_power: int = 1
    n_oversamples: int = 8
    random_state: int = 42

    def __post_init__(self) -> None:
        if self.n_components <= 0:
            raise ValueError("n_components must be positive")
        if self.iterated_power < 0 or self.n_oversamples <= 0:
            raise ValueError("PCA randomized-SVD settings are invalid")


class XGBoostPCAClassifier:
    """Fit PCA and XGBoost on training samples, then transform other splits.

    Call :meth:`fit` with training data only. Validation and test data should
    be passed only to :meth:`transform` or the prediction methods.
    """

    def __init__(
        self,
        pca_config: PCAConfig | None = None,
        xgboost_config: XGBoostConfig | None = None,
    ) -> None:
        self.pca_config = pca_config or PCAConfig()
        self.xgboost_config = xgboost_config or XGBoostConfig()
        self.pca = PCA(
            n_components=self.pca_config.n_components,
            svd_solver="randomized",
            iterated_power=self.pca_config.iterated_power,
            n_oversamples=self.pca_config.n_oversamples,
            random_state=self.pca_config.random_state,
        )
        self.classifier = XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            n_estimators=self.xgboost_config.n_estimators,
            max_depth=self.xgboost_config.max_depth,
            learning_rate=self.xgboost_config.learning_rate,
            subsample=self.xgboost_config.subsample,
            colsample_bytree=self.xgboost_config.colsample_bytree,
            random_state=self.xgboost_config.random_state,
            n_jobs=self.xgboost_config.n_jobs,
            tree_method="hist",
            device="cpu",
            verbosity=0,
        )
        self._fitted = False
        self.training_predictions_: np.ndarray | None = None
        self.training_probabilities_: np.ndarray | None = None
        self.pca_fit_seconds_ = 0.0
        self.xgboost_fit_seconds_ = 0.0

    @staticmethod
    def _validate_features(features: np.ndarray, name: str) -> None:
        if features.ndim != 2 or features.shape[0] == 0:
            raise ValueError(f"{name} must be a non-empty 2D feature matrix")
        if not np.isfinite(features).all():
            raise ValueError(f"{name} contains non-finite values")

    def fit(self, train_features: np.ndarray, train_labels: np.ndarray) -> XGBoostPCAClassifier:
        """Fit PCA and XGBoost using training samples only."""
        self._validate_features(train_features, "train_features")
        if train_labels.ndim != 1 or train_labels.shape[0] != train_features.shape[0]:
            raise ValueError("train_labels must match train_features")
        if set(np.unique(train_labels).tolist()) != {0, 1}:
            raise ValueError("Training labels must contain both binary classes 0 and 1")
        if train_features.shape[0] < self.pca_config.n_components:
            raise ValueError("Training sample count is smaller than PCA components")
        if train_features.shape[1] < self.pca_config.n_components:
            raise ValueError("Feature dimension is smaller than PCA components")

        pca_started = perf_counter()
        with threadpool_limits(limits=4):
            self.pca.fit(train_features)
        self.pca_fit_seconds_ = perf_counter() - pca_started
        with threadpool_limits(limits=4):
            train_pca = self.pca.transform(train_features)
        xgboost_started = perf_counter()
        self.classifier.fit(train_pca, train_labels)
        self.xgboost_fit_seconds_ = perf_counter() - xgboost_started
        self._fitted = True
        self.training_predictions_ = self.classifier.predict(train_pca).astype(
            np.int8
        )
        self.training_probabilities_ = self.classifier.predict_proba(train_pca)
        return self

    def transform(self, features: np.ndarray) -> np.ndarray:
        """Apply the already-fitted PCA without fitting or changing it."""
        if not self._fitted:
            raise RuntimeError("Fit the classifier on training data before transform")
        self._validate_features(features, "features")
        with threadpool_limits(limits=4):
            return self.pca.transform(features)

    def predict_transformed(self, features: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Classifier has not been fitted")
        return self.classifier.predict(features).astype(np.int8)

    def predict_proba_transformed(self, features: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Classifier has not been fitted")
        return self.classifier.predict_proba(features)

    def predict(self, features: np.ndarray) -> np.ndarray:
        return self.predict_transformed(self.transform(features))

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        return self.predict_proba_transformed(self.transform(features))

    def configuration(self) -> dict[str, Any]:
        return {
            "pca": {
                **asdict(self.pca_config),
                "method": "PCA randomized SVD",
                "fit_split": "train only",
            },
            "xgboost": {
                **asdict(self.xgboost_config),
                "objective": "binary:logistic",
                "eval_metric": "logloss",
                "tree_method": "hist",
                "device": "cpu",
            },
        }
