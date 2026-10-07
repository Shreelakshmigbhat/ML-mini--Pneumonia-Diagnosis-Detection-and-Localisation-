"""Random Forest baseline for flattened normalized chest X-ray images."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.validation import check_is_fitted


@dataclass(frozen=True)
class RandomForestConfig:
    """Deterministic bounded-depth forest configuration."""

    n_estimators: int = 100
    criterion: str = "gini"
    max_depth: int = 20
    max_features: str = "sqrt"
    min_samples_leaf: int = 2
    bootstrap: bool = True
    class_weight: str | None = None
    random_state: int = 42
    n_jobs: int = -1

    def __post_init__(self) -> None:
        if self.n_estimators <= 0:
            raise ValueError("n_estimators must be positive")
        if self.criterion not in {"gini", "entropy", "log_loss"}:
            raise ValueError(f"Unsupported criterion: {self.criterion}")
        if self.max_depth <= 0:
            raise ValueError("max_depth must be positive")
        if self.max_features not in {"sqrt", "log2"}:
            raise ValueError("max_features must be 'sqrt' or 'log2'")
        if self.min_samples_leaf <= 0:
            raise ValueError("min_samples_leaf must be positive")
        if self.n_jobs == 0:
            raise ValueError("n_jobs cannot be zero")


class RandomForestBaseline:
    """Small API wrapper around sklearn's deterministic RandomForestClassifier."""

    def __init__(self, config: RandomForestConfig | None = None) -> None:
        self.config = config or RandomForestConfig()
        self.estimator = RandomForestClassifier(**asdict(self.config))

    def fit(self, features: Any, labels: Any) -> RandomForestBaseline:
        self.estimator.fit(features, labels)
        return self

    def predict(self, features: Any) -> Any:
        check_is_fitted(self.estimator)
        return self.estimator.predict(features)

    def score(self, features: Any, labels: Any) -> float:
        check_is_fitted(self.estimator)
        return float(self.estimator.score(features, labels))

    def save(self, path: Path) -> None:
        check_is_fitted(self.estimator)
        import joblib

        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {"estimator": self.estimator, "configuration": asdict(self.config)},
            path,
            compress=3,
        )

    def configuration(self) -> dict[str, Any]:
        return asdict(self.config)
