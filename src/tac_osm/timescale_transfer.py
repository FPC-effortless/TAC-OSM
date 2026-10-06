"""Fixed temporal-scale transfer mechanism for TDBU research.

The only novel mechanism in this lane is the choice of a distributed bank of
fixed exponential traces. The surrounding policy, executor, outcome and
verifier are conventional scaffolding reused from the MTSK lane.
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np

from tac_osm.mtsk_topdown import (
    FixedBinaryExecutor,
    IndependentVerifier,
    MLPPolicy,
    environment_outcome,
)

EXPERIMENT_ID = "TACOSM-PLM-TDBU-TIMESCALE-TRANSFER-001"
GENERATOR_VERSION = "timescale-transfer-v1-train-test-disjoint-0.30-0.992"
BENCHMARK_HASH = hashlib.sha256(GENERATOR_VERSION.encode()).hexdigest()

TRAIN_FILTER_ALPHAS = np.array(
    (0.30, 0.50, 0.68, 0.82, 0.90, 0.96, 0.985),
    dtype=np.float64,
)
TEST_FILTER_ALPHAS = np.array(
    (0.40, 0.58, 0.74, 0.86, 0.93, 0.975, 0.992),
    dtype=np.float64,
)
DISTRIBUTED_EIGHT_ALPHAS = np.array(
    (0.33, 0.47, 0.62, 0.78, 0.875, 0.935, 0.98, 0.995),
    dtype=np.float64,
)
ARM_ALPHAS = {
    "no_state": np.array((), dtype=np.float64),
    "one_timescale": np.array((0.90,), dtype=np.float64),
    "two_timescale": np.array((0.50, 0.98), dtype=np.float64),
    "three_timescale": np.array((0.50, 0.90, 0.98), dtype=np.float64),
    "distributed_eight": DISTRIBUTED_EIGHT_ALPHAS,
}
ARMS = tuple(ARM_ALPHAS)
Arm = Literal[
    "no_state",
    "one_timescale",
    "two_timescale",
    "three_timescale",
    "distributed_eight",
]
PAIR_TYPES_TRAIN = tuple(
    (i, j)
    for i in range(len(TRAIN_FILTER_ALPHAS))
    for j in range(i + 1, len(TRAIN_FILTER_ALPHAS))
)
PAIR_TYPES_TEST = tuple(
    (i, j)
    for i in range(len(TEST_FILTER_ALPHAS))
    for j in range(i + 1, len(TEST_FILTER_ALPHAS))
)
INTERPOLATION_TEST_INDICES = tuple(range(5))
EXTRAPOLATION_TEST_INDICES = (5, 6)
POLICY_HIDDEN = {
    "no_state": 12,
    "one_timescale": 16,
    "two_timescale": 14,
    "three_timescale": 12,
    "distributed_eight": 7,
}


def _filter(history: np.ndarray, alpha: float) -> float:
    state = 0.0
    for value in history:
        state = alpha * state + (1.0 - alpha) * float(value)
    return state


def target_action(
    history: np.ndarray,
    pair_type: tuple[int, int],
    alphas: np.ndarray,
) -> int:
    left = _filter(history, float(alphas[pair_type[0]]))
    right = _filter(history, float(alphas[pair_type[1]]))
    return int(left >= right)


@dataclass(frozen=True)
class EpisodeExample:
    history: np.ndarray
    label: int
    pair_id: int
    current_observation: float = 0.0
    filter_family: int = 0


def make_pairs(
    seed: int,
    history_length: int,
    n_pairs: int,
    *,
    alphas: np.ndarray,
    pair_types: Sequence[tuple[int, int]],
) -> list[EpisodeExample]:
    if history_length < 1 or n_pairs < 1:
        raise ValueError("history_length and n_pairs must be positive")
    rng = np.random.default_rng(seed)
    examples: list[EpisodeExample] = []
    for pair_id in range(n_pairs):
        pair_type = pair_types[pair_id % len(pair_types)]
        history = rng.normal(0.0, 1.0, size=history_length).astype(np.float64)
        family = pair_type[0] * len(alphas) + pair_type[1]
        if target_action(history, pair_type, alphas) != 1:
            history = -history
        examples.append(EpisodeExample(history.copy(), 1, pair_id, 0.0, family))
        examples.append(EpisodeExample(-history, 0, pair_id, 0.0, family))
    return examples


def split_filter_subset(
    examples: list[EpisodeExample],
    subset_indices: Sequence[int],
    *,
    alphas: np.ndarray,
    pair_types: Sequence[tuple[int, int]],
) -> list[EpisodeExample]:
    allowed = set(int(i) for i in subset_indices)
    wanted_pairs = {
        pid % len(pair_types)
        for pid in range(len(pair_types) * len(allowed))
    }
    out: list[EpisodeExample] = []
    for ex in examples:
        pt = pair_types[ex.pair_id % len(pair_types)]
        if pt[0] in allowed and pt[1] in allowed:
            out.append(ex)
    if not out:
        raise ValueError("filter subset produced no examples")
    return out


class FixedTemporalState:
    """Internal distributed exponential state bank for one registered arm."""

    def __init__(self, arm: Arm):
        self.arm = arm
        self.alphas = ARM_ALPHAS[arm]
        self.values = np.zeros(len(self.alphas), dtype=np.float64)

    @property
    def footprint(self) -> int:
        return int(len(self.values))

    def reset(self) -> None:
        self.values.fill(0.0)

    def process(self, history: np.ndarray) -> np.ndarray:
        self.reset()
        for observation in history:
            x = float(observation)
            for i, alpha in enumerate(self.alphas):
                a = float(alpha)
                self.values[i] = a * self.values[i] + (1.0 - a) * x
        return self.values.copy()


def _features_for_state(
    examples: list[EpisodeExample],
    arm: Arm,
    *,
    intervention: str = "normal",
) -> np.ndarray:
    state = FixedTemporalState(arm)
    rows: list[np.ndarray] = []
    for example in examples:
        final = state.process(example.history)
        rows.append(np.concatenate(([example.current_observation], final)))
    features = np.asarray(rows, dtype=np.float64)
    if intervention == "reset":
        features[:, 1:] = 0.0
    elif intervention == "shuffle":
        features[:, 1:] = features[np.arange(len(features))[::-1], 1:]
    elif intervention != "normal":
        raise ValueError(f"unknown intervention {intervention!r}")
    return features


def make_policy(seed: int, arm: Arm) -> MLPPolicy:
    return MLPPolicy(seed=seed, hidden=POLICY_HIDDEN[arm])


def parameter_count(arm: Arm) -> int:
    d = 1 + len(ARM_ALPHAS[arm])
    h = POLICY_HIDDEN[arm]
    return int(d * h + h + h * 2 + 2)


def evaluation_work(arm: Arm, history_length: int) -> float:
    k = len(ARM_ALPHAS[arm])
    state_update = history_length * 3 * k
    router_macs = (1 + k) * POLICY_HIDDEN[arm] + 2 * POLICY_HIDDEN[arm]
    verifier = 1
    return float(state_update + router_macs + verifier)


@dataclass(frozen=True)
class Evaluation:
    success: float
    action_gap: float
    state_footprint: int
    evaluation_work_per_episode: float


def evaluate(
    policy: MLPPolicy,
    examples: list[EpisodeExample],
    arm: Arm,
    *,
    intervention: str = "normal",
) -> Evaluation:
    features = _features_for_state(examples, arm, intervention=intervention)
    predictions = policy.predict(features)
    labels = np.asarray([x.label for x in examples], dtype=np.int64)
    executor = FixedBinaryExecutor()
    verifier = IndependentVerifier()
    executed = np.asarray(
        [executor.execute(int(a)) for a in predictions], dtype=np.int64
    )
    outcomes = np.asarray(
        [
            environment_outcome(int(a), int(y))
            for a, y in zip(executed, labels)
        ],
        dtype=np.float64,
    )
    verified = np.asarray(
        [
            verifier.verify(int(a), int(y))
            for a, y in zip(executed, labels)
        ],
        dtype=np.float64,
    )
    if not np.array_equal(outcomes, verified):
        raise RuntimeError("environment outcome and verifier disagree")

    pair_predictions: dict[int, list[int]] = {}
    for example, prediction in zip(examples, predictions):
        pair_predictions.setdefault(example.pair_id, []).append(int(prediction))
    action_gap = float(
        np.mean(
            [
                len(values) == 2 and values[0] != values[1]
                for values in pair_predictions.values()
            ]
        )
    )
    return Evaluation(
        success=float(np.mean(verified)),
        action_gap=action_gap,
        state_footprint=len(ARM_ALPHAS[arm]),
        evaluation_work_per_episode=evaluation_work(
            arm, len(examples[0].history)
        ),
    )


def representability_witness(history_length: int = 256) -> dict[str, float]:
    """Impulse-response R² of each held-out temporal filter in the 8-trace bank."""
    t = np.arange(history_length, dtype=np.float64)
    basis = np.stack(
        [(1.0 - a) * (a**t) for a in DISTRIBUTED_EIGHT_ALPHAS],
        axis=1,
    )
    result: dict[str, float] = {}
    for index, alpha in enumerate(TEST_FILTER_ALPHAS):
        target = (1.0 - alpha) * (alpha**t)
        coefficients, *_ = np.linalg.lstsq(basis, target, rcond=None)
        prediction = basis @ coefficients
        denominator = np.sum((target - target.mean()) ** 2)
        r2 = 1.0 - np.sum((target - prediction) ** 2) / denominator
        result[f"test_alpha_{index}"] = float(r2)
    return result


def benchmark_hash() -> str:
    return BENCHMARK_HASH
