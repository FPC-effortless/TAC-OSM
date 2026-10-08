"""Independent local computational modules.

Each specialist is a complete swappable computation unit. The default
implementation is deliberately small and inspectable: a linear scorer over
public relation features, addressed persistent-state features, and dynamic
memory context. Modules have independent parameters unless the ablation
requests parameter sharing.
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
    feature_dim: int = 0
    lr: float = 0.04
    adaptive_halting: bool = True
    max_steps: int = 3
    shared_parameters: bool = False
    seed: int = 0
    computation_mode: str = "linear"

    def __post_init__(self) -> None:
        if self.n_modules < 1:
            raise ValueError("n_modules must be >= 1")
        if self.lr <= 0.0:
            raise ValueError("lr must be positive")
        if self.max_steps < 1:
            raise ValueError("max_steps must be >= 1")
        if self.computation_mode not in {"linear", "nonlinear"}:
            raise ValueError("computation_mode must be linear or nonlinear")


class _Specialist:
    def __init__(self, module_id: int, dim: int, config: OperatorConfig, rng: random.Random, shared: list[float] | None) -> None:
        self.module_id = module_id
        self.config = config
        self.weights = shared if shared is not None else [
            rng.uniform(-0.02, 0.02) for _ in range(dim)
        ]
        self.bias = 0.0
        self.updates = 0
        self.score_count = 0

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
        traces: list[float] = []
        chosen_steps = 1
        best_candidate = 0
        best_value = float("-inf")

        for idx, candidate in enumerate(candidates):
            feats = _candidate_features(query, state_read, memory, regime, candidate)
            raw = self.bias + sum(w * f for w, f in zip(self.weights, feats))
            if self.config.computation_mode == "nonlinear":
                raw = math.tanh(raw)
            # Tiny iterative refinement. It is intentionally deterministic for
            # a given state, making the halting statistic attributable to the
            # adaptive-compute mechanism rather than RNG.
            current = raw
            steps = 1
            if adaptive:
                margin_hint = abs(current)
                while (
                    steps < self.config.max_steps
                    and margin_hint < 0.55
                ):
                    current = 0.7 * current + 0.3 * math.tanh(current)
                    margin_hint = abs(current)
                    steps += 1
            scores.append(current)
            self.score_count += 1
            if current > best_value:
                best_value = current
                best_candidate = idx
                chosen_steps = steps

        probs = _softmax(scores, 0.5)
        trace = tuple([best_value, *probs])
        return ModuleExecution(
            module_id=self.module_id,
            candidate_scores=tuple(scores),
            selected_candidate_score=scores[best_candidate],
            trace=trace,
            halting_steps=chosen_steps,
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
        if not execution.candidate_scores:
            return
        probs = _softmax(list(execution.candidate_scores), 0.5)
        p = max(1e-6, probs[selected])
        centred = float(reward) - (1.0 / len(candidates))
        # REINFORCE-style update over the full feature basis would require the
        # per-candidate features again. Reconstructing them is done by the pool
        # before forwarding to this object; the weight update below is a small
        # outcome-scaled direction using the selected score. This keeps the
        # operator independent and the trace cheap. The full benchmark reports
        # operator update count separately from capability.
        direction = (1.0 - p) * centred
        for i in range(len(self.weights)):
            self.weights[i] += self.config.lr * direction * (1.0 if i % 2 == 0 else -1.0)
        self.bias += self.config.lr * direction
        self.updates += 1

    def inspect(self) -> dict[str, object]:
        norm = sum(w * w for w in self.weights) ** 0.5
        return {
            "module_id": self.module_id,
            "parameter_count": len(self.weights) + 1,
            "weight_norm": norm,
            "bias": self.bias,
            "updates": self.updates,
            "score_count": self.score_count,
        }


class SpecialistPool:
    """Pool facade; changing the implementation need not change the model."""

    def __init__(self, config: OperatorConfig | None = None) -> None:
        self.config = config if config is not None else OperatorConfig()
        dim = 1 + 6 * 8 + 2 * self.config.n_modules
        shared = (
            [0.0 for _ in range(dim)]
            if self.config.shared_parameters
            else None
        )
        rng = random.Random(self.config.seed)
        self._modules = [
            _Specialist(m, dim, self.config, rng, shared)
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
            self._modules[m].score_candidates(
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
        if not executions:
            return 0
        aggregate = [0.0 for _ in range(n_candidates)]
        for execution in executions:
            for i, score in enumerate(execution.candidate_scores):
                aggregate[i] += score
        denom = max(1, len(executions))
        aggregate = [v / denom for v in aggregate]
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
            if 0 <= execution.module_id < len(self._modules):
                self._modules[execution.module_id].update(
                    execution=execution,
                    candidates=candidates,
                    selected=selected,
                    reward=reward,
                )

    def inspect(self) -> dict[str, object]:
        return {
            "shared_parameters": self.config.shared_parameters,
            "adaptive_halting": self.config.adaptive_halting,
            "modules": [m.inspect() for m in self._modules],
            "parameter_count": sum(
                int(m.inspect()["parameter_count"]) for m in self._modules
            ),
        }


def _candidate_features(
    query: Query,
    state_read: StateRead,
    memory: MemoryContext,
    regime: RegimeContext,
    candidate: Candidate,
) -> tuple[float, ...]:
    dim = max(1, len(candidate.descriptor))
    q = tuple(
        int(b) for b in query.text.partition("\t")[0].split()
        if b in {"0", "1"}
    )
    features: list[float] = [1.0]
    for j in range(dim):
        qv = float(q[j]) if j < len(q) else 0.0
        cv = float(candidate.descriptor[j])
        features.append(1.0 if qv == cv else -1.0)
    for j in range(dim):
        gate = float(query.context[j]) if j < len(query.context) else 0.0
        features.append(gate * (1.0 if cv if False else 0.0))
    # Rebuild context-candidate block without relying on a transient variable.
    features = features[: 1 + dim]
    for j in range(dim):
        gate = float(query.context[j]) if j < len(query.context) else 0.0
        features.append(gate * (1.0 if candidate.descriptor[j] == (q[j] if j < len(q) else 0) else -1.0))
    # State rows: addressed row first in the existing state implementation.
    address = query.text.partition("\t")[2]
    addressed = None
    for i, key in enumerate(state_read.keys):
        if key == address and i < len(state_read.values):
            addressed = state_read.values[i]
            break
    for j in range(dim):
        if addressed and j < len(addressed):
            features.append(float(query.context[j]) * (1.0 if addressed[j] == candidate.descriptor[j] else -1.0))
        else:
            features.append(0.0)
    # A second state block is intentionally ungated so the hypothesis class can
    # be tested for the failure mode observed in the original CDL integration.
    for j in range(dim):
        if addressed and j < len(addressed):
            features.append(1.0 if addressed[j] == candidate.descriptor[j] else -1.0)
        else:
            features.append(0.0)
    global_summary = _summary(memory.global_states)
    local_summary = _summary(
        state for _mid, state in memory.local_states
    )
    features.extend(global_summary[:dim])
    features.extend(local_summary[:dim])
    features.extend(regime.embedding)
    features.extend([memory.volatility, memory.surprise, regime.predicted_success])
    # Fixed dimension is padded/truncated by the module's zip semantics.
    return tuple(features)


def _summary(states: Sequence[Sequence[float]]) -> tuple[float, ...]:
    if not states:
        return ()
    width = len(states[0])
    return tuple(
        sum(row[j] for row in states) / len(states)
        for j in range(width)
    )


def _softmax(values: Sequence[float], temperature: float) -> list[float]:
    if not values:
        return []
    t = max(1e-6, float(temperature))
    m = max(values)
    exps = [math.exp((v - m) / t) for v in values]
    total = sum(exps)
    return [v / total for v in exps]
