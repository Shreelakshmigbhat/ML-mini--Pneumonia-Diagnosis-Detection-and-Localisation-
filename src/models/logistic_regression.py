"""Binary logistic regression trained with a NumPy-only L-BFGS optimizer."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class LogisticRegressionConfig:
    l2_strength: float = 1e-4
    max_iterations: int = 100
    tolerance: float = 1e-5
    validation_interval: int = 5
    line_search_max_steps: int = 30
    history_size: int = 10
    armijo_constant: float = 1e-4

    def __post_init__(self) -> None:
        if self.l2_strength < 0:
            raise ValueError("l2_strength must be non-negative")
        if self.max_iterations <= 0:
            raise ValueError("max_iterations must be positive")
        if self.tolerance <= 0:
            raise ValueError("tolerance must be positive")
        if self.validation_interval <= 0:
            raise ValueError("validation_interval must be positive")
        if self.line_search_max_steps <= 0 or self.history_size <= 0:
            raise ValueError("line search and history sizes must be positive")
        if not 0 < self.armijo_constant < 1:
            raise ValueError("armijo_constant must be between 0 and 1")


class LogisticRegression:
    """L2-regularized binary logistic regression for flattened image vectors."""

    def __init__(self, config: LogisticRegressionConfig | None = None) -> None:
        self.config = config or LogisticRegressionConfig()
        self.coef_: np.ndarray | None = None
        self.intercept_: float | None = None
        self.best_iteration_: int | None = None
        self.best_validation_accuracy_: float | None = None
        self.validation_history_: list[dict[str, float | int]] = []

    def _objective(
        self, features: np.ndarray, labels: np.ndarray, parameters: np.ndarray
    ) -> float:
        weights = parameters[:-1]
        intercept = parameters[-1]
        logits = features @ weights + intercept
        loss = np.logaddexp(0.0, logits) - labels * logits
        return float(
            np.mean(loss, dtype=np.float64)
            + 0.5 * self.config.l2_strength * np.dot(weights, weights)
        )

    def _objective_and_gradient(
        self, features: np.ndarray, labels: np.ndarray, parameters: np.ndarray
    ) -> tuple[float, np.ndarray]:
        weights = parameters[:-1]
        intercept = parameters[-1]
        logits = features @ weights + intercept
        probabilities = np.exp(-np.logaddexp(0.0, -logits))
        residuals = probabilities - labels
        sample_count = labels.size

        loss = np.logaddexp(0.0, logits) - labels * logits
        objective = float(
            np.mean(loss, dtype=np.float64)
            + 0.5 * self.config.l2_strength * np.dot(weights, weights)
        )
        weight_gradient = (
            features.T @ residuals / sample_count
            + self.config.l2_strength * weights
        )
        gradient = np.empty_like(parameters)
        gradient[:-1] = weight_gradient
        gradient[-1] = np.mean(residuals, dtype=np.float64)
        return objective, gradient

    @staticmethod
    def _validate_arrays(
        features: np.ndarray, labels: np.ndarray, name: str
    ) -> None:
        if features.ndim != 2:
            raise ValueError(f"{name} features must be 2D, got {features.shape}")
        if labels.ndim != 1 or labels.shape[0] != features.shape[0]:
            raise ValueError(
                f"{name} labels must align with features; "
                f"got {features.shape} and {labels.shape}"
            )
        if not features.shape[0] or not features.shape[1]:
            raise ValueError(f"{name} data must not be empty")
        if not np.isfinite(features).all() or not np.isfinite(labels).all():
            raise ValueError(f"{name} data contains non-finite values")
        if not np.isin(labels, (0, 1)).all():
            raise ValueError(f"{name} labels must be binary values 0 and 1")

    def fit(
        self,
        train_features: np.ndarray,
        train_labels: np.ndarray,
        validation_features: np.ndarray,
        validation_labels: np.ndarray,
    ) -> LogisticRegression:
        self._validate_arrays(train_features, train_labels, "train")
        self._validate_arrays(validation_features, validation_labels, "validation")
        if train_features.shape[1] != validation_features.shape[1]:
            raise ValueError("Train and validation feature dimensions do not match")
        if set(np.unique(train_labels)) != {0, 1}:
            raise ValueError("Train split must contain both binary classes")
        if set(np.unique(validation_labels)) != {0, 1}:
            raise ValueError("Validation split must contain both binary classes")

        features = np.asarray(train_features, dtype=np.float64, order="C")
        labels = np.asarray(train_labels, dtype=np.float64)
        validation_features = np.asarray(validation_features, dtype=np.float64)
        validation_labels = np.asarray(validation_labels, dtype=np.int8)
        parameters = np.zeros(features.shape[1] + 1, dtype=np.float64)
        objective, gradient = self._objective_and_gradient(
            features, labels, parameters
        )

        s_history: list[np.ndarray] = []
        y_history: list[np.ndarray] = []
        inverse_curvature: list[float] = []
        best_parameters: np.ndarray | None = None
        best_accuracy = -1.0
        best_iteration = 0
        last_checkpoint_iteration = -1
        completed_iterations = 0

        def checkpoint(iteration: int) -> None:
            nonlocal best_parameters, best_accuracy, best_iteration
            logits = validation_features @ parameters[:-1] + parameters[-1]
            accuracy = float(
                np.mean((logits >= 0.0).astype(np.int8) == validation_labels)
            )
            self.validation_history_.append(
                {
                    "iteration": iteration,
                    "accuracy": accuracy,
                    "objective": objective,
                }
            )
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_parameters = parameters.copy()
                best_iteration = iteration

        for iteration in range(1, self.config.max_iterations + 1):
            if np.linalg.norm(gradient, ord=np.inf) <= self.config.tolerance:
                break

            q = gradient.copy()
            alphas: list[float] = []
            for s_vector, y_vector, rho in zip(
                reversed(s_history),
                reversed(y_history),
                reversed(inverse_curvature),
            ):
                alpha = rho * float(np.dot(s_vector, q))
                alphas.append(alpha)
                q -= alpha * y_vector

            if s_history:
                last_s = s_history[-1]
                last_y = y_history[-1]
                scale = float(np.dot(last_s, last_y) / np.dot(last_y, last_y))
            else:
                scale = 1.0
            inverse_hessian_product = scale * q
            for (s_vector, y_vector, rho), alpha in zip(
                zip(s_history, y_history, inverse_curvature),
                reversed(alphas),
            ):
                beta = rho * float(np.dot(y_vector, inverse_hessian_product))
                inverse_hessian_product += s_vector * (alpha - beta)
            direction = -inverse_hessian_product

            directional_derivative = float(np.dot(gradient, direction))
            if not np.isfinite(directional_derivative) or directional_derivative >= 0:
                direction = -gradient
                directional_derivative = -float(np.dot(gradient, gradient))

            step_size = 1.0
            accepted = False
            for _ in range(self.config.line_search_max_steps):
                candidate = parameters + step_size * direction
                candidate_objective = self._objective(
                    features, labels, candidate
                )
                if (
                    np.isfinite(candidate_objective)
                    and candidate_objective
                    <= objective
                    + self.config.armijo_constant
                    * step_size
                    * directional_derivative
                ):
                    accepted = True
                    break
                step_size *= 0.5
            if not accepted:
                raise RuntimeError(
                    f"L-BFGS line search failed at iteration {iteration}"
                )

            candidate_objective, candidate_gradient = self._objective_and_gradient(
                features, labels, candidate
            )
            s_vector = candidate - parameters
            y_vector = candidate_gradient - gradient
            curvature = float(np.dot(s_vector, y_vector))
            curvature_floor = (
                1e-12
                * np.linalg.norm(s_vector)
                * np.linalg.norm(y_vector)
            )
            if curvature > curvature_floor:
                if len(s_history) == self.config.history_size:
                    s_history.pop(0)
                    y_history.pop(0)
                    inverse_curvature.pop(0)
                s_history.append(s_vector)
                y_history.append(y_vector)
                inverse_curvature.append(1.0 / curvature)

            parameters = candidate
            objective = candidate_objective
            gradient = candidate_gradient
            completed_iterations = iteration

            if (
                iteration % self.config.validation_interval == 0
                or iteration == self.config.max_iterations
            ):
                checkpoint(iteration)
                last_checkpoint_iteration = iteration

        if completed_iterations != last_checkpoint_iteration:
            checkpoint(completed_iterations)

        if best_parameters is None:
            raise RuntimeError(
                "Optimization ended before a validation checkpoint was evaluated"
            )
        self.coef_ = best_parameters[:-1].copy()
        self.intercept_ = float(best_parameters[-1])
        self.best_iteration_ = best_iteration
        self.best_validation_accuracy_ = best_accuracy
        return self

    def decision_function(self, features: np.ndarray) -> np.ndarray:
        if self.coef_ is None or self.intercept_ is None:
            raise RuntimeError("Model has not been fitted")
        if features.ndim != 2 or features.shape[1] != self.coef_.size:
            raise ValueError("Features do not match the fitted model dimensions")
        return features @ self.coef_ + self.intercept_

    def predict(self, features: np.ndarray) -> np.ndarray:
        return (self.decision_function(features) >= 0.0).astype(np.int8)

    def score(self, features: np.ndarray, labels: np.ndarray) -> float:
        if labels.ndim != 1 or labels.shape[0] != features.shape[0]:
            raise ValueError("Labels do not match feature rows")
        return float(np.mean(self.predict(features) == labels))

    def save(self, path: str) -> None:
        if self.coef_ is None or self.intercept_ is None:
            raise RuntimeError("Cannot save a model before fitting")
        np.savez_compressed(
            path,
            coefficients=self.coef_,
            intercept=np.asarray(self.intercept_),
            best_iteration=np.asarray(self.best_iteration_),
            best_validation_accuracy=np.asarray(
                self.best_validation_accuracy_
            ),
        )

    def configuration(self) -> dict[str, Any]:
        return asdict(self.config)
