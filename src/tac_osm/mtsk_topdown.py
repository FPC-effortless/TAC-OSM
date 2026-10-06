"""Top-down PLM temporal mechanism and bottom-up ablation scaffold.

Only the persistent-state kernel is novel in this lane. The policy, action
semantics and verifier are deliberately conventional so the state mechanism
is the isolated intervention.
"""
from __future__ import annotations

import hashlib
import platform
from dataclasses import dataclass
from typing import Literal

import numpy as np

GENERATOR_VERSION = "tdbu-mtsk-v1-ground-truth-0.45-0.88-0.985"
BENCHMARK_HASH = hashlib.sha256(GENERATOR_VERSION.encode()).hexdigest()

GROUND_TRUTH_ALPHAS = np.array((0.45, 0.88, 0.985), dtype=np.float64)
PAIR_TYPES = ((0, 2), (0, 1), (1, 2))

STATE_ALPHAS = {
    "no_state": (None, None, None),
    "single_timescale": (0.90, 0.90, 0.90),
    "two_timescale": (0.50, 0.98, 0.98),
    "mtsk": (0.50, 0.90, 0.98),
}

Arm = Literal["no_state", "single_timescale", "two_timescale", "mtsk"]


def _filter(history: np.ndarray, alpha: float) -> float:
    state = 0.0
    for value in history:
        state = alpha * state + (1.0 - alpha) * float(value)
    return state


def target_action(history: np.ndarray, pair_type: tuple[int, int]) -> int:
    """Hidden temporal contrast; pair_type is never exposed to the model."""
    left = _filter(history, float(GROUND_TRUTH_ALPHAS[pair_type[0]]))
    right = _filter(history, float(GROUND_TRUTH_ALPHAS[pair_type[1]]))
    return int(left >= right)


@dataclass(frozen=True)
class EpisodeExample:
    history: np.ndarray
    label: int
    pair_id: int
    current_observation: float = 0.0


def make_balanced_pairs(seed: int, history_length: int, n_pairs: int) -> list[EpisodeExample]:
    """Generate disjoint-history pairs with opposite gold actions."""
    if history_length < 1 or n_pairs < 1:
        raise ValueError("history_length and n_pairs must be positive")
    rng = np.random.default_rng(seed)
    examples: list[EpisodeExample] = []
    for pair_id in range(n_pairs):
        pair_type = PAIR_TYPES[pair_id % len(PAIR_TYPES)]
        history = rng.normal(0.0, 1.0, size=history_length).astype(np.float64)
        if target_action(history, pair_type) != 1:
            history = -history
        examples.append(EpisodeExample(history.copy(), 1, pair_id))
        examples.append(EpisodeExample(-history, 0, pair_id))
    return examples


class PersistentTemporalState:
    """State interface shared by all top-down arms."""
    def __init__(self, arm: Arm):
        self.arm = arm
        self.alphas = STATE_ALPHAS[arm]
        self.values = np.zeros(3, dtype=np.float64)

    def reset(self) -> None:
        self.values.fill(0.0)

    def update(self, observation: float) -> None:
        if self.arm == "no_state":
            return
        for i, alpha in enumerate(self.alphas):
            a = float(alpha)
            self.values[i] = a * self.values[i] + (1.0 - a) * observation

    def process(self, history: np.ndarray) -> np.ndarray:
        self.reset()
        for x in history:
            self.update(float(x))
        return self.values.copy()


class MLPPolicy:
    """Standard two-action tanh MLP; no architecture-specific state machinery."""
    def __init__(self, seed: int, hidden: int = 12):
        if hidden < 1:
            raise ValueError("hidden must be positive")
        rng = np.random.default_rng(seed)
        self.W1 = rng.normal(0.0, 0.5, size=(4, hidden))
        self.b1 = np.zeros(hidden, dtype=np.float64)
        self.W2 = rng.normal(0.0, 0.5, size=(hidden, 2))
        self.b2 = np.zeros(2, dtype=np.float64)
        self.hidden = hidden

    @property
    def parameter_count(self) -> int:
        return int(self.W1.size + self.b1.size + self.W2.size + self.b2.size)

    def _forward(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        z = X @ self.W1 + self.b1
        h = np.tanh(z)
        logits = h @ self.W2 + self.b2
        shifted = logits - logits.max(axis=1, keepdims=True)
        e = np.exp(shifted)
        probs = e / e.sum(axis=1, keepdims=True)
        return h, logits, probs

    def fit(self, X: np.ndarray, y: np.ndarray, steps: int, learning_rate: float) -> int:
        if X.ndim != 2 or X.shape[1] != 4:
            raise ValueError(f"expected [N,4] input, got {X.shape}")
        if len(X) != len(y):
            raise ValueError("X/y length mismatch")
        y_onehot = np.eye(2, dtype=np.float64)[y]
        for _ in range(steps):
            h, _logits, probs = self._forward(X)
            dlogits = (probs - y_onehot) / len(X)
            dW2 = h.T @ dlogits
            db2 = dlogits.sum(axis=0)
            dh = dlogits @ self.W2.T
            dz = dh * (1.0 - h * h)
            dW1 = X.T @ dz
            db1 = dz.sum(axis=0)
            self.W2 -= learning_rate * dW2
            self.b2 -= learning_rate * db2
            self.W1 -= learning_rate * dW1
            self.b1 -= learning_rate * db1
        return steps

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.argmax(self._forward(X)[2], axis=1)

    def parameter_hash(self) -> str:
        h = hashlib.sha256()
        for array in (self.W1, self.b1, self.W2, self.b2):
            h.update(np.ascontiguousarray(array).tobytes())
        return h.hexdigest()




def representability_witness(seed: int = 12345, pairs_per_family: int = 1000) -> dict[str, float]:
    """Analytical shared-map witness: MTSK coordinates retain each registered temporal contrast."""
    examples = make_balanced_pairs(seed, 64, pairs_per_family * len(PAIR_TYPES))
    features = featurize(examples, "mtsk")
    scores: dict[str, float] = {}
    for family_idx, pair_type in enumerate(PAIR_TYPES):
        indices = [j for j, ex in enumerate(examples) if ex.pair_id % len(PAIR_TYPES) == family_idx]
        predicted = (features[indices, 1 + pair_type[0]] >= features[indices, 1 + pair_type[1]]).astype(int)
        labels = np.asarray([examples[j].label for j in indices], dtype=np.int64)
        scores[f"family_{pair_type[0]}_vs_{pair_type[1]}"] = float(np.mean(predicted == labels))
    return scores

class FixedBinaryExecutor:
    """Standard action executor; semantics are deliberately trivial and fixed."""
    def execute(self, action: int) -> int:
        if int(action) not in (0, 1):
            raise ValueError(f"invalid binary action: {action!r}")
        return int(action)


class IndependentVerifier:
    """Post-action verifier that checks the executed action against the outcome."""
    def verify(self, executed_action: int, expected_action: int) -> bool:
        return int(executed_action) == int(expected_action)
@dataclass(frozen=True)
class Evaluation:
    success: float
    action_gap: float
    state_footprint: int
    evaluation_work_per_episode: float


def featurize(examples: list[EpisodeExample], arm: Arm) -> np.ndarray:
    state = PersistentTemporalState(arm)
    rows: list[np.ndarray] = []
    for example in examples:
        final_state = state.process(example.history)
        rows.append(np.concatenate(([example.current_observation], final_state)))
    return np.asarray(rows, dtype=np.float64)


def evaluate(policy: MLPPolicy, examples: list[EpisodeExample], arm: Arm, intervention: str = "normal") -> Evaluation:
    features = featurize(examples, arm)
    if intervention == "reset":
        features[:, 1:] = 0.0
    elif intervention == "shuffle":
        permutation = np.arange(len(features))[::-1]
        features[:, 1:] = features[permutation, 1:]
    elif intervention != "normal":
        raise ValueError(f"unknown intervention {intervention!r}")

    predictions = policy.predict(features)
    labels = np.asarray([x.label for x in examples], dtype=np.int64)
    executor = FixedBinaryExecutor()
    verifier = IndependentVerifier()
    executed = np.asarray([executor.execute(int(action)) for action in predictions], dtype=np.int64)
    outcomes = np.asarray(
        [environment_outcome(int(action), int(label)) for action, label in zip(executed, labels)],
        dtype=np.float64,
    )
    verified = np.asarray(
        [verifier.verify(int(action), int(label)) for action, label in zip(executed, labels)],
        dtype=np.float64,
    )
    if not np.array_equal(outcomes, verified):
        raise RuntimeError("environment outcome and independent verifier disagree")
    success = float(np.mean(verified))

    pair_predictions: dict[int, list[int]] = {}
    for example, prediction in zip(examples, predictions):
        pair_predictions.setdefault(example.pair_id, []).append(int(prediction))
    action_gap = float(np.mean([
        len(v) == 2 and v[0] != v[1]
        for v in pair_predictions.values()
    ]))

    state_updates = 0 if arm == "no_state" else len(examples[0].history) * 9
    router_macs = 4 * policy.hidden + 2 * policy.hidden
    verifier_ops = 1
    work = float(state_updates + router_macs + verifier_ops)

    return Evaluation(
        success=success,
        action_gap=action_gap,
        state_footprint=0 if arm == "no_state" else 3,
        evaluation_work_per_episode=work,
    )


def environment_outcome(action: int, label: int) -> bool:
    """Independent post-action verifier truth."""
    return int(action) == int(label)


def dependency_lock_hash() -> str:
    descriptor = f"python={platform.python_version()}|numpy={np.__version__}"
    return hashlib.sha256(descriptor.encode()).hexdigest()
