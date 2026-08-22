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

    def __init__(self, feature_names: Sequence[str]) -> None:
        if not feature_names:
            raise ValueError("At least one distance feature is required")
        self.feature_names = list(feature_names)
        self.logits_ = np.zeros(len(self.feature_names), dtype=float)

    @property
    def weights_(self) -> np.ndarray:
        return softmax(self.logits_)

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
            return self.pairwise_loss(positive, negative)

        result = minimize(objective, self.logits_, method="BFGS",
                          options={"maxiter": max_iter, "gtol": 1e-8})
        self.logits_ = np.asarray(result.x, float)
        return self

    def save(self, path: Path) -> None:
        payload = {"feature_names": self.feature_names,
                   "weights": dict(zip(self.feature_names, map(float, self.weights_))),
                   "logits": self.logits_.tolist()}
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "LinearPairwiseRanker":
        payload = json.loads(path.read_text(encoding="utf-8"))
        model = cls(payload["feature_names"])
        model.logits_ = np.asarray(payload["logits"], float)
        return model
