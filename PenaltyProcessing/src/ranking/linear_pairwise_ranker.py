"""Non-negative softmax-weighted linear distance trained from ranking pairs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np
from scipy.optimize import minimize


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = np.asarray(logits, float) - np.max(logits)
    exp = np.exp(shifted)
    return exp / exp.sum()


class LinearPairwiseRanker:
    """Learn component budgets; smaller returned scores are better."""

    def __init__(
        self, feature_names: Sequence[str], *, regularization_strength: float = 0.0,
        minimum_weight: float = 0.0, prior_weights: Sequence[float] | None = None,
    ) -> None:
        if not feature_names:
            raise ValueError("At least one distance feature is required")
        self.feature_names = list(feature_names)
        if not np.isfinite(regularization_strength) or regularization_strength < 0:
            raise ValueError("regularization_strength must be finite and non-negative")
        if not np.isfinite(minimum_weight) or minimum_weight < 0:
            raise ValueError("minimum_weight must be finite and non-negative")
        if minimum_weight * len(self.feature_names) >= 1.0:
            raise ValueError("minimum_weight leaves no probability mass to learn")
        self.regularization_strength = float(regularization_strength)
        self.minimum_weight = float(minimum_weight)
        if prior_weights is None:
            self.prior_weights_ = np.full(
                len(self.feature_names), 1.0 / len(self.feature_names), dtype=float,
            )
        else:
            prior = np.asarray(prior_weights, float)
            if prior.shape != (len(self.feature_names),) or not np.isfinite(prior).all() \
                    or np.any(prior < 0) or prior.sum() <= 0:
                raise ValueError("prior_weights must be finite, non-negative, and match features")
            self.prior_weights_ = prior / prior.sum()
        self.logits_ = np.zeros(len(self.feature_names), dtype=float)

    @property
    def weights_(self) -> np.ndarray:
        free_mass = 1.0 - self.minimum_weight * len(self.feature_names)
        return self.minimum_weight + free_mass * softmax(self.logits_)

    def regularization_penalty(self) -> float:
        """Squared distance from an interpretable prior over component budgets."""
        return float(self.regularization_strength * np.sum(
            np.square(self.weights_ - self.prior_weights_)
        ))

    def score(self, distances: np.ndarray) -> np.ndarray:
        x = np.asarray(distances, float)
        one_dimensional = x.ndim == 1
        if one_dimensional:
            x = x[None, :]
        available = np.isfinite(x)
        weighted = np.where(available, x, 0.0) @ self.weights_
        mass = available @ self.weights_
        result = np.divide(weighted, mass, out=np.full(len(x), np.inf), where=mass > 0)
        return result[0] if one_dimensional else result

    def pairwise_loss(self, positive: np.ndarray, negative: np.ndarray) -> float:
        margin = self.score(positive) - self.score(negative)
        return float(np.mean(np.logaddexp(0.0, margin)))

    def fit(
        self, positive: np.ndarray, negative: np.ndarray, *, max_iter: int = 500,
    ) -> "LinearPairwiseRanker":
        positive, negative = np.asarray(positive, float), np.asarray(negative, float)
        if positive.shape != negative.shape or positive.ndim != 2:
            raise ValueError("positive and negative must be equal-shaped 2-D arrays")
        if positive.shape[1] != len(self.feature_names):
            raise ValueError("Pair width does not match feature_names")

        def objective(logits: np.ndarray) -> float:
            self.logits_ = logits
            return self.pairwise_loss(positive, negative) + self.regularization_penalty()

        result = minimize(objective, self.logits_, method="BFGS",
                          options={"maxiter": max_iter, "gtol": 1e-8})
        self.logits_ = np.asarray(result.x, float)
        # Softmax is invariant to a shared offset. Centering prevents saved
        # logits from drifting to unnecessarily large values.
        self.logits_ -= self.logits_.mean()
        return self

    def save(self, path: Path) -> None:
        payload = {"feature_names": self.feature_names,
                   "weights": dict(zip(self.feature_names, map(float, self.weights_))),
                   "logits": self.logits_.tolist(),
                   "regularization_strength": self.regularization_strength,
                   "minimum_weight": self.minimum_weight,
                   "prior_weights": dict(zip(
                       self.feature_names, map(float, self.prior_weights_)
                   ))}
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "LinearPairwiseRanker":
        payload = json.loads(path.read_text(encoding="utf-8"))
        names = payload["feature_names"]
        prior_payload = payload.get("prior_weights")
        prior = ([prior_payload[name] for name in names]
                 if isinstance(prior_payload, dict) else None)
        model = cls(
            names,
            regularization_strength=float(payload.get("regularization_strength", 0.0)),
            minimum_weight=float(payload.get("minimum_weight", 0.0)),
            prior_weights=prior,
        )
        model.logits_ = np.asarray(payload["logits"], float)
        return model
