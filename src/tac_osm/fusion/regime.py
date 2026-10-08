"""Learned slow regime/condition state.

The estimator is a tiny online predictor. It learns from prior outcomes only;
the current hidden target never enters its input. Its output is an adaptive
condition state, not a hand-written task label.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Sequence

from .. import Outcome
from .interfaces import RegimeContext


@dataclass(frozen=True)
class RegimeConfig:
    enabled: bool = True
    feature_dim: int = 16
    embedding_dim: int = 4
    lr: float = 0.02
    seed: int = 0
    novelty_decay: float = 0.97

    def __post_init__(self) -> None:
        if self.feature_dim < 1 or self.embedding_dim < 1:
            raise ValueError("feature_dim and embedding_dim must be positive")
        if self.lr <= 0.0:
            raise ValueError("lr must be positive")
        if not 0.0 < self.novelty_decay < 1.0:
            raise ValueError("novelty_decay must be in (0,1)")


class RegimeState:
    """Online outcome predictor plus a compact latent regime representation."""

    def __init__(self, config: RegimeConfig | None = None) -> None:
        self.config = config if config is not None else RegimeConfig()
        rng = random.Random(self.config.seed)
        self._w = [
            [rng.uniform(-0.05, 0.05) for _ in range(self.config.feature_dim)]
            for _ in range(self.config.embedding_dim)
        ]
        self._predictor = [
            rng.uniform(-0.05, 0.05) for _ in range(self.config.feature_dim)
        ]
        self._bias = 0.0
        self._novelty_ema = 1.0
        self._last_features: tuple[float, ...] | None = None
        self._last_prediction = 0.5
        self._updates = 0

    def context(self, features: Sequence[float]) -> RegimeContext:
        if not self.config.enabled:
            return RegimeContext((), 0.5, 0.0, 0.0, 0.0)

        x = _fit(features, self.config.feature_dim)
        pred = _sigmoid(
            self._bias + sum(a * b for a, b in zip(self._predictor, x))
        )
        embedding = tuple(
            math.tanh(sum(a * b for a, b in zip(row, x)))
            for row in self._w
        )
        novelty = (
            1.0
            if self._last_features is None
            else _distance(x, self._last_features)
        )
        self._novelty_ema = (
            self.config.novelty_decay * self._novelty_ema
            + (1.0 - self.config.novelty_decay) * novelty
        )
        # Confidence is high when the model makes a decisive prediction.
        confidence = min(1.0, abs(pred - 0.5) * 2.0)
        volatility = min(1.0, self._novelty_ema)

        self._last_features = x
        self._last_prediction = pred
        return RegimeContext(
            embedding=embedding,
            predicted_success=pred,
            volatility=volatility,
            novelty=novelty,
            confidence=confidence,
        )

    def update(
        self,
        *,
        features: Sequence[float],
        outcome: Outcome,
        reward: float,
    ) -> None:
        if not self.config.enabled:
            return
        x = _fit(features, self.config.feature_dim)
        target = 1.0 if outcome.success else 0.0
        err = target - self._last_prediction
        signed_reward = float(reward if reward != 0 else (1.0 if outcome.success else -1.0))
        step = self.config.lr * signed_reward
        for j, value in enumerate(x):
            self._predictor[j] += step * err * value
        self._bias += step * err
        for row in self._w:
            for j, value in enumerate(x):
                row[j] += 0.25 * step * err * value
        self._updates += 1

    def inspect(self) -> dict[str, object]:
        return {
            "enabled": self.config.enabled,
            "updates": self._updates,
            "embedding_dim": self.config.embedding_dim,
            "predicted_success": self._last_prediction,
            "novelty": self._novelty_ema,
            "confidence": min(1.0, abs(self._last_prediction - 0.5) * 2.0),
        }


def _fit(values: Sequence[float], dim: int) -> tuple[float, ...]:
    return tuple(
        float(values[i]) if i < len(values) else 0.0 for i in range(dim)
    )


def _distance(a: Sequence[float], b: Sequence[float]) -> float:
    if not a or not b:
        return 0.0
    return min(
        1.0,
        sum((x - y) ** 2 for x, y in zip(a, b)) ** 0.5
        / max(1.0, len(a) ** 0.5),
    )


def _sigmoid(x: float) -> float:
    if x >= 40:
        return 1.0
    if x <= -40:
        return 0.0
    return 1.0 / (1.0 + math.exp(-x))
