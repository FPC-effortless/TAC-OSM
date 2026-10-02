from dataclasses import dataclass

import pytest

from tac_osm.unified import (
    Budget,
    Execution,
    MemoryWrite,
    Observation,
    Plan,
    PredictiveState,
    ProposalSet,
    Structure,
    UnifiedPersistentStructuralWorldModel,
    Verification,
)


class D:
    def discover(self, observation, knowledge):
        return ()


class A:
    def address(self, state, knowledge, budget):
        return ProposalSet(
            candidates=(Structure("a", (1.0,), "op"),),
            recall_upper_bound=1.0,
            work_units=1,
        )


class P:
    def predict(self, state, structure, depth=1):
        return (1.0,)


class PlanCtl:
    def plan(self, state, proposals, predictor, budget):
        return Plan(proposals.candidates, ((1.0,),), 0, 1)


class E:
    def __init__(self):
        self.calls = 0

    def execute(self, plan, state):
        self.calls += 1
        return Execution("a", (float(self.calls),), ((float(self.calls),),), 1)


class V:
    def __init__(self):
        self.calls = 0

    def verify(self, state, execution, observation):
        self.calls += 1
        return Verification(
            valid=self.calls > 1,
            confidence=1.0,
            failed_constraint="bad" if self.calls == 1 else "",
        )


class R:
    def repair(self, state, plan, verification, attempts_used, budget):
        return plan if attempts_used == 1 else None


class W:
    def __init__(self):
        self.calls = 0
        self.verified = []

    def decide(self, state, execution, verification, budget):
        self.calls += 1
        self.verified.append(verification.valid)
        return (
            (
                MemoryWrite(
                    "add",
                    "a",
                    execution.output,
                    1.0,
                    "verified",
                ),
            )
            if verification.valid
            else ()
        )


def _model(executor=None, verifier=None, writer=None, repairer=None):
    return UnifiedPersistentStructuralWorldModel(
        discoverer=D(),
        addressor=A(),
        predictor=P(),
        planner=PlanCtl(),
        executor=executor or E(),
        verifier=verifier or V(),
        writer=writer or W(),
        repairer=repairer or R(),
        budget=Budget(max_repair_attempts=2),
    )


def test_verification_precedes_write_and_repair_is_bounded():
    executor = E()
    verifier = V()
    writer = W()
    model = _model(executor=executor, verifier=verifier, writer=writer)
    state = PredictiveState((0.0,), ("future",), 1.0)
    obs = Observation("language", (0.0,), "train:0", 0)

    _next, execution, verification = model.step(obs, state)

    assert executor.calls == 2
    assert verifier.calls == 2
    assert writer.calls == 1
    assert writer.verified == [True]
    assert verification.valid
    assert model.history[-1]["repair_attempts"] == 1
    assert len(model.store.knowledge.experiences) == 1
    assert model.store.knowledge.experiences[0].verification.valid


def test_proposal_miss_does_not_fallback_to_exhaustive():
    class EmptyAddressor:
        def address(self, state, knowledge, budget):
            return ProposalSet((), 0.0, 1)

    model = _model()
    model.addressor = EmptyAddressor()

    with pytest.raises(RuntimeError, match="fallback"):
        model.step(
            Observation("audio", (0.0,), "train:0", 0),
            PredictiveState((0.0,), (), 1.0),
        )
