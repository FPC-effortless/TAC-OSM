"""Single-relation temporal-memory benchmark with explicit sequential state."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

from tac_osm.mtsk_topdown import (
    FixedBinaryExecutor,
    IndependentVerifier,
    MLPPolicy,
    environment_outcome,
)

EXPERIMENT_ID = "TACOSM-PLM-TDBU-MEMORY-ISOLATION-001"
GENERATOR_VERSION = "memory-isolation-v2-online-state-ground-truth-0.45-0.985"
BENCHMARK_HASH = hashlib.sha256(GENERATOR_VERSION.encode()).hexdigest()
GROUND_TRUTH_ALPHAS = (0.45, 0.985)

STATE_ALPHAS = {
    "no_state": (None, None, None),
    "single_timescale": (0.90, None, None),
    "two_timescale": (0.50, 0.98, None),
    "mtsk": (0.50, 0.90, 0.98),
}
ARMS = ("no_state", "single_timescale", "two_timescale", "mtsk")


@dataclass(frozen=True)
class EpisodeExample:
    history: np.ndarray
    label: int
    pair_id: int
    current_observation: float = 0.0


class PersistentTemporalState:
    """Explicit sequential state update used by every stateful arm."""

    def __init__(self, arm: str):
        if arm not in ARMS:
            raise ValueError(f"unknown arm: {arm!r}")
        self.arm = arm
        self.alphas = STATE_ALPHAS[arm]
        self.values = np.zeros(3, dtype=np.float64)

    def reset(self) -> None:
        self.values.fill(0.0)

    def update(self, observation: float) -> None:
        if self.arm == "no_state":
            return
        for i, alpha in enumerate(self.alphas):
            if alpha is not None:
                a = float(alpha)
                self.values[i] = a * self.values[i] + (1.0 - a) * float(observation)

    def process(self, history: np.ndarray) -> np.ndarray:
        self.reset()
        for observation in history:
            self.update(float(observation))
        return self.values.copy()


def _filter(history: np.ndarray, alpha: float) -> float:
    state = 0.0
    for value in history:
        state = alpha * state + (1.0 - alpha) * float(value)
    return state


def target_action(history: np.ndarray) -> int:
    return int(_filter(history, GROUND_TRUTH_ALPHAS[0]) >= _filter(history, GROUND_TRUTH_ALPHAS[1]))


def make_pairs(seed: int, history_length: int, n_pairs: int) -> list[EpisodeExample]:
    if history_length < 1 or n_pairs < 1:
        raise ValueError("history_length and n_pairs must be positive")
    rng = np.random.default_rng(seed)
    examples: list[EpisodeExample] = []

    for pair_id in range(n_pairs):
        history = rng.normal(0.0, 1.0, size=history_length).astype(np.float64)
        if target_action(history) != 1:
            history = -history
        examples.append(EpisodeExample(history.copy(), 1, pair_id))
        examples.append(EpisodeExample(-history, 0, pair_id))

    return examples


def featurize(examples: list[EpisodeExample], arm: str) -> np.ndarray:
    """Produce the policy input by running the state recurrence observation-by-observation."""
    state = PersistentTemporalState(arm)
    rows = []
    for example in examples:
        terminal_state = state.process(example.history)
        rows.append(
            np.concatenate(
                (
                    np.asarray([example.current_observation], dtype=np.float64),
                    terminal_state,
                )
            )
        )
    return np.asarray(rows, dtype=np.float64)


def representability_sign_accuracy(seed: int = 12345, n_pairs: int = 3000) -> float:
    """Independent witness that MTSK state coordinates retain the fixed contrast."""
    examples = make_pairs(seed, 64, n_pairs)
    X = featurize(examples, "mtsk")
    y = np.asarray([e.label for e in examples], dtype=np.int64)
    prediction = (X[:, 1] >= X[:, 3]).astype(int)
    return float(np.mean(prediction == y))


def evaluate(
    policy: MLPPolicy,
    examples: list[EpisodeExample],
    arm: str,
    intervention: str = "normal",
) -> dict[str, float]:
    X = featurize(examples, arm)

    if intervention == "reset":
        X[:, 1:] = 0.0
    elif intervention == "shuffle":
        X[:, 1:] = X[np.arange(len(X))[::-1], 1:]
    elif intervention != "normal":
        raise ValueError(intervention)

    prediction = policy.predict(X)
    labels = np.asarray([example.label for example in examples], dtype=np.int64)

    executor = FixedBinaryExecutor()
    verifier = IndependentVerifier()
    actions = np.asarray([executor.execute(int(action)) for action in prediction], dtype=np.int64)
    outcomes = np.asarray(
        [environment_outcome(int(action), int(target)) for action, target in zip(actions, labels)],
        dtype=np.float64,
    )
    verified = np.asarray(
        [verifier.verify(int(action), int(target)) for action, target in zip(actions, labels)],
        dtype=np.float64,
    )

    if not np.array_equal(outcomes, verified):
        raise RuntimeError("environment outcome and independent verifier disagree")

    pair_predictions: dict[int, list[int]] = {}
    for example, prediction_value in zip(examples, prediction):
        pair_predictions.setdefault(example.pair_id, []).append(int(prediction_value))

    action_gap = float(
        np.mean(
            [len(values) == 2 and values[0] != values[1] for values in pair_predictions.values()]
        )
    )
    state_updates_per_episode = len(examples[0].history) * (
        0 if arm == "no_state" else sum(alpha is not None for alpha in STATE_ALPHAS[arm])
    )
    router_macs = 4 * policy.hidden + 2 * policy.hidden
    verifier_ops = 1
    work = float(state_updates_per_episode + router_macs + verifier_ops)

    return {
        "success": float(np.mean(verified)),
        "action_gap": action_gap,
        "work": work,
    }


def benchmark_hash() -> str:
    return BENCHMARK_HASH


def dependency_hash() -> str:
    return hashlib.sha256(b"numpy-only-memory-isolation-v2").hexdigest()
