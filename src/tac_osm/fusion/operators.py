"""Independent local computational modules.

Each specialist is a complete swappable computation unit. The default
implementation is a small outcome-trained linear operator with an optional
deterministic refinement loop. Every module owns its parameters and cached
feature trace, so training, inspection, replacement, and lesions remain local.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Sequence

from .. import Candidate, Query, StateRead
from .interfaces import MemoryContext, ModuleExecution, RegimeContext


@dataclass(frozen=True)
class OperatorConfig:
    n_modules: int = 8
    dim: int = 8
    lr: float = 0.04
    adaptive_halting: bool = True
    max_steps: int = 3
    shared_parameters: bool = False
    seed: int = 0
    computation_mode: str = "linear"
    temperature: float = 0.5

    def __post_init__(self) -> None:
        if self.n_modules < 1:
            raise ValueError("n_modules must be >= 1")
        if self.dim < 1:
            raise ValueError("dim must be >= 1")
        if self.lr <= 0.0:
            raise ValueError("lr must be positive")
        if self.max_steps < 1:
            raise ValueError("max_steps must be >= 1")
        if self.computation_mode not in {"linear", "nonlinear"}:
            raise ValueError("computation_mode must be linear or nonlinear")
        if self.temperature <= 0.0:
            raise ValueError("temperature must be positive")


class _Specialist:
    def __init__(
        self,
        module_id: int,
        feature_dim: int,
        config: OperatorConfig,
        rng: random.Random,
        shared: list[float] | None,
    ) -> None:
        self.module_id = module_id
        self.config = config
        self.weights = (
            shared if shared is not None
            else [rng.uniform(-0.02, 0.02) for _ in range(feature_dim)]
        )
        self.bias = 0.0
        self.updates = 0
        self.score_count = 0
        self._last_features: tuple[tuple[float, ...], ...] = ()

    def score_candidates(
        self,
        *,
        query: Query,
        state_read: StateRead,
        memory: MemoryContext,
        regime: RegimeContext,
        candidates: Sequence[Candidate],
        adaptive: bool,
    ) -> ModuleExecution:
        scores: list[float] = []
        cached: list[tuple[float, ...]] = []

        for candidate in candidates:
            feats = _fit(
                _candidate_features(query, state_read, memory, regime, candidate),
                len(self.weights),
            )
            cached.append(feats)
            raw = self.bias + sum(w * f for w, f in zip(self.weights, feats))
            current = math.tanh(raw) if self.config.computation_mode == "nonlinear" else raw
            steps = 1
            while adaptive and steps < self.config.max_steps and abs(current) < 0.55:
                current = 0.7 * current + 0.3 * math.tanh(current)
                steps += 1
            scores.append(current)
            self.score_count += 1

        self._last_features = tuple(cached)
        best_candidate = max(range(len(scores)), key=lambda i: (scores[i], -i))
        probs = _softmax(scores, self.config.temperature)
        return ModuleExecution(
            module_id=self.module_id,
            candidate_scores=tuple(scores),
            selected_candidate_score=scores[best_candidate],
            trace=tuple([scores[best_candidate], *probs]),
            halting_steps=_halting_steps(scores, adaptive, self.config.max_steps),
            provenance=f"specialist:{self.module_id}",
        )

    def update(
        self,
        *,
        execution: ModuleExecution,
        candidates: Sequence[Candidate],
        selected: int,
        reward: float,
    ) -> None:
        if not self._last_features or not candidates:
            return
        if not 0 <= selected < len(self._last_features):
            return
        probs = _softmax(list(execution.candidate_scores), self.config.temperature)
        p = max(1e-6, probs[selected])
        centred = float(reward) - (1.0 / len(candidates))
        direction = (1.0 - p) * centred
        for i, feature in enumerate(self._last_features[selected]):
            self.weights[i] += self.config.lr * direction * feature
        self.bias += self.config.lr * direction
        self.updates += 1

    def inspect(self) -> dict[str, object]:
        return {
            "module_id": self.module_id,
            "parameter_count": len(self.weights) + 1,
            "weight_norm": sum(w * w for w in self.weights) ** 0.5,
            "bias": self.bias,
            "updates": self.updates,
            "score_count": self.score_count,
            "feature_dimension": len(self.weights),
            "last_trace_length": len(self._last_features),
        }


class SpecialistPool:
    """Pool facade; module implementations can be replaced independently."""

    def __init__(self, config: OperatorConfig | None = None) -> None:
        self.config = config if config is not None else OperatorConfig()
        feature_dim = _feature_dimension(self.config.dim)
        shared = [0.0] * feature_dim if self.config.shared_parameters else None
        rng = random.Random(self.config.seed)
        self._modules = [
            _Specialist(m, feature_dim, self.config, rng, shared)
            for m in range(self.config.n_modules)
        ]

    @property
    def modules(self) -> tuple[_Specialist, ...]:
        return tuple(self._modules)

    def execute_modules(
        self,
        *,
        selected_modules: Sequence[int],
        query: Query,
        state_read: StateRead,
        memory: MemoryContext,
        regime: RegimeContext,
        candidates: Sequence[Candidate],
    ) -> tuple[ModuleExecution, ...]:
        return tuple(
            self._modules[int(m)].score_candidates(
                query=query,
                state_read=state_read,
                memory=memory,
                regime=regime,
                candidates=candidates,
                adaptive=self.config.adaptive_halting,
            )
            for m in selected_modules
            if 0 <= int(m) < len(self._modules)
        )

    def select_candidate(
        self,
        executions: Sequence[ModuleExecution],
        n_candidates: int,
    ) -> int:
        if not executions or n_candidates < 1:
            return 0
        aggregate = [0.0] * n_candidates
        counts = [0] * n_candidates
        for execution in executions:
            for i, score in enumerate(execution.candidate_scores[:n_candidates]):
                aggregate[i] += score
                counts[i] += 1
        aggregate = [
            aggregate[i] / max(1, counts[i]) for i in range(n_candidates)
        ]
        return max(range(n_candidates), key=lambda i: (aggregate[i], -i))

    def update(
        self,
        *,
        executions: Sequence[ModuleExecution],
        candidates: Sequence[Candidate],
        selected: int,
        reward: float,
    ) -> None:
        for execution in executions:
            module_id = int(execution.module_id)
            if 0 <= module_id < len(self._modules):
                self._modules[module_id].update(
                    execution=execution,
                    candidates=candidates,
                    selected=selected,
                    reward=reward,
                )

    def disable(self, module_id: int) -> None:
        """Hard lesion; parameters remain inspectable after the lesion."""
        if not 0 <= module_id < len(self._modules):
            raise IndexError(module_id)
        module = self._modules[module_id]
        module.weights[:] = [0.0] * len(module.weights)
        module.bias = -1e9

    def inspect(self) -> dict[str, object]:
        pairwise = []
        for i, left in enumerate(self._modules):
            for right in self._modules[i + 1:]:
                pairwise.append(
                    sum((a - b) ** 2 for a, b in zip(left.weights, right.weights)) ** 0.5
                )
        logical_parameter_count = (
            len(self._modules[0].weights) + 1
            if self.config.shared_parameters else
            sum(int(m.inspect()["parameter_count"]) for m in self._modules)
        )
        return {
            "shared_parameters": self.config.shared_parameters,
            "adaptive_halting": self.config.adaptive_halting,
            "modules": [m.inspect() for m in self._modules],
            "parameter_count": logical_parameter_count,
            "mean_pairwise_parameter_distance": (
                sum(pairwise) / len(pairwise) if pairwise else 0.0
            ),
        }


def _feature_dimension(dim: int) -> int:
    return 1 + 7 * dim + 7


def _candidate_features(
    query: Query,
    state_read: StateRead,
    memory: MemoryContext,
    regime: RegimeContext,
    candidate: Candidate,
) -> tuple[float, ...]:
    dim = len(candidate.descriptor)
    q = tuple(
        int(b) for b in query.text.partition("\t")[0].split()
        if b in {"0", "1"}
    )
    features: list[float] = [1.0]

    for j in range(dim):
        qj = q[j] if j < len(q) else 0
        features.append(1.0 if qj == candidate.descriptor[j] else -1.0)

    for j in range(dim):
        gate = float(query.context[j]) if j < len(query.context) else 0.0
        qj = q[j] if j < len(q) else 0
        features.append(gate * (1.0 if candidate.descriptor[j] == qj else -1.0))

    address = query.text.partition("\t")[2]
    addressed: Sequence[int] = ()
    for i, key in enumerate(state_read.keys):
        if key == address and i < len(state_read.values):
            addressed = state_read.values[i]
            break

    for j in range(dim):
        if j < len(addressed):
            gate = float(query.context[j]) if j < len(query.context) else 0.0
            features.append(gate * (1.0 if addressed[j] == candidate.descriptor[j] else -1.0))
        else:
            features.append(0.0)

    # Ungated address match is retained deliberately as a testable hypothesis
    # class; the representability gate can expose whether it harms composition.
    for j in range(dim):
        features.append(
            1.0 if j < len(addressed) and addressed[j] == candidate.descriptor[j]
            else 0.0
        )

    global_summary = _summary(memory.global_states)
    local_summary = _summary([state for _mid, state in memory.local_states])
    for j in range(dim):
        features.append(global_summary[j] if j < len(global_summary) else 0.0)
    for j in range(dim):
        features.append(local_summary[j] if j < len(local_summary) else 0.0)

    for j in range(4):
        features.append(regime.embedding[j] if j < len(regime.embedding) else 0.0)
    features.extend([memory.volatility, memory.surprise, regime.predicted_success])
    return tuple(features)


def _summary(states: Sequence[Sequence[float]]) -> tuple[float, ...]:
    if not states:
        return ()
    width = len(states[0])
    return tuple(sum(row[j] for row in states) / len(states) for j in range(width))


def _fit(values: Sequence[float], dim: int) -> tuple[float, ...]:
    if len(values) >= dim:
        return tuple(float(v) for v in values[:dim])
    return tuple(float(v) for v in values) + (0.0,) * (dim - len(values))


def _softmax(values: Sequence[float], temperature: float) -> list[float]:
    if not values:
        return []
    t = max(1e-6, float(temperature))
    m = max(values)
    exps = [math.exp((v - m) / t) for v in values]
    total = sum(exps) or 1.0
    return [v / total for v in exps]


def _halting_steps(scores: Sequence[float], adaptive: bool, max_steps: int) -> int:
    if not adaptive or not scores:
        return 1
    if len(scores) == 1:
        return 1 if abs(scores[0]) >= 0.55 else min(max_steps, 2)
    ordered = sorted(scores, reverse=True)
    margin = ordered[0] - ordered[1]
    return 1 if margin >= 0.55 else min(max_steps, 2)
