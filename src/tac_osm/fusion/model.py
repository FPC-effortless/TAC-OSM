"""Full end-to-end fused TAC-OSM model.

The orchestration boundary is deliberately narrow. Every mechanism is injected:
memory, regime estimator, coordinator, specialist pool, verifier and repair
policy can be replaced independently. The environment and persistent state are
also injected, so the same model can be used with synthetic, multimodal, or
embodied environments later without changing the control loop.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .. import Candidate, Environment, Outcome, PersistentState, StateUpdate
from .interfaces import FusionStep
from .memory import MemoryConfig, MultiScaleMemory
from .regime import RegimeConfig, RegimeState
from .coordinator import CoordinatorConfig, SparseCoordinator
from .operators import OperatorConfig, SpecialistPool
from .verifier import FusionVerifier, FusionVerifierConfig, RepairPolicy


@dataclass(frozen=True)
class FusionModelConfig:
    n_steps: int = 24
    seed: int = 0
    learning: bool = True
    experience_write: bool = True
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    regime: RegimeConfig = field(default_factory=RegimeConfig)
    coordinator: CoordinatorConfig = field(default_factory=CoordinatorConfig)
    operators: OperatorConfig = field(default_factory=OperatorConfig)
    verifier: FusionVerifierConfig = field(default_factory=FusionVerifierConfig)
    repair_enabled: bool = True

    def __post_init__(self) -> None:
        if self.n_steps < 1:
            raise ValueError("n_steps must be >=1")


class FusedTacOsmModel:
    """PNDS with distributed local circuits and multi-timescale dynamics."""

    def __init__(
        self,
        *,
        state: PersistentState,
        environment: Environment,
        config: FusionModelConfig | None = None,
        memory: MultiScaleMemory | None = None,
        regime: RegimeState | None = None,
        coordinator: SparseCoordinator | None = None,
        operators: SpecialistPool | None = None,
        verifier: FusionVerifier | None = None,
        repair: RepairPolicy | None = None,
    ) -> None:
        self.config = config if config is not None else FusionModelConfig()
        self.state = state
        self.environment = environment
        self.memory = memory if memory is not None else MultiScaleMemory(self.config.memory)
        self.regime = regime if regime is not None else RegimeState(self.config.regime)
        self.coordinator = coordinator if coordinator is not None else SparseCoordinator(self.config.coordinator)
        self.operators = operators if operators is not None else SpecialistPool(self.config.operators)
        self.verifier = verifier if verifier is not None else FusionVerifier(self.config.verifier)
        self.repair = repair if repair is not None else RepairPolicy(enabled=self.config.repair_enabled)
        self.steps: list[FusionStep] = []

    def step(self, step_index: int) -> FusionStep:
        task = self.environment.next_task(self.state)
        query = task.public()
        read = self.state.read(query)

        # The regime is generated strictly from current observable inputs plus
        # dynamic state. The hidden target remains on Task/environment only.
        base_features = _observable_features(query, read)
        regime = self.regime.context(base_features)
        memory = self.memory.context()

        coordination = self.coordinator.route(
            query=query,
            state_read=read,
            memory=memory,
            regime=regime,
        )
        # Local state becomes available only for the selected modules, matching
        # the local-loop abstraction. This also makes local-vs-global ablations
        # causally meaningful.
        memory = self.memory.context(coordination.selected_modules)
        executions = self.operators.execute_modules(
            selected_modules=coordination.selected_modules,
            query=query,
            state_read=read,
            memory=memory,
            regime=regime,
            candidates=task.candidates,
        )
        selected = self.operators.select_candidate(executions, len(task.candidates))

        # O_t boundary: the outcome is revealed only after the complete routing
        # and computation path has finished.
        outcome = self.environment.transition(self.state, selected, query)
        passed, feedback = self.verifier.verify(
            execution=executions,
            selected=selected,
            candidates=task.candidates,
            outcome=outcome,
        )
        repair = self.repair.propose(passed=passed, feedback=feedback)

        reward = 1.0 if outcome.success else 0.0
        surprise = 1.0 - regime.predicted_success if outcome.success else regime.predicted_success

        # Dynamic memory is experience state, not authoritative world state. It
        # updates after observation and can therefore only influence future steps.
        self.memory.observe(
            module_ids=coordination.selected_modules,
            signal=base_features,
            reward=reward,
            surprise=surprise,
            success=outcome.success,
        )
        self.regime.update(features=base_features, outcome=outcome, reward=reward)

        if self.config.learning:
            self.coordinator.update(decision=coordination, reward=reward)
            self.operators.update(
                executions=executions,
                candidates=task.candidates,
                selected=selected,
                reward=reward,
            )

        wrote = False
        if self.config.experience_write and self.verifier.commit_allowed(
            passed=passed,
            outcome=outcome,
        ):
            key = f"experience:{task.family}:{step_index}"
            value = tuple(int(b) for b in task.candidates[selected].descriptor)
            write = self.state.write(
                StateUpdate(
                    key=key,
                    value=value,
                    task_key=task.family,
                    success_score=reward,
                    step=step_index,
                )
            )
            wrote = bool(write.committed)

        result = FusionStep(
            step=step_index,
            family=task.family,
            query=query,
            state_read=read,
            memory=memory,
            regime=regime,
            coordination=coordination,
            module_executions=executions,
            selected_candidate=selected,
            outcome=outcome,
            verification_passed=passed,
            verification_feedback=feedback,
            repair=repair,
            wrote_experience=wrote,
            provenance={
                "model": "tacosm_fused_bioinspired_2026-10-08",
                "router_sees_target": False,
                "environment_transition_after_route": True,
                "experience_write_requires_verified_success": (
                    self.verifier.config.mode != "none"
                ),
            },
        )
        self.steps.append(result)
        return result

    def run(self) -> tuple[FusionStep, ...]:
        self.steps.clear()
        return tuple(self.step(i) for i in range(self.config.n_steps))

    def inspect(self) -> dict[str, Any]:
        return {
            "config": _dataclass_tree(self.config),
            "memory": self.memory.inspect(),
            "regime": self.regime.inspect(),
            "coordinator": self.coordinator.inspect(),
            "operators": self.operators.inspect(),
            "verifier": self.verifier.config.__dict__.copy(),
            "repair": {
                "enabled": self.repair.enabled,
                "max_attempts": self.repair.max_attempts,
            },
            "steps": len(self.steps),
        }


def _observable_features(query: Any, read: Any) -> tuple[float, ...]:
    bits = tuple(
        int(b) for b in query.text.partition("\t")[0].split()
        if b in {"0", "1"}
    )
    out = [1.0]
    out.extend(1.0 if b else -1.0 for b in bits[:8])
    out.extend(float(x) for x in query.context[:8])
    out.append(min(1.0, len(read.values) / 16.0))
    out.append(1.0 if query.text.partition("\t")[2] else 0.0)
    return tuple(out)


def _dataclass_tree(value: Any) -> Any:
    if hasattr(value, "__dataclass_fields__"):
        return {
            key: _dataclass_tree(getattr(value, key))
            for key in value.__dataclass_fields__
        }
    if isinstance(value, tuple):
        return [_dataclass_tree(v) for v in value]
    if isinstance(value, list):
        return [_dataclass_tree(v) for v in value]
    return value
