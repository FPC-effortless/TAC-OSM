"""DEV-016: action-dependent delayed maintenance, standard-library validity gate.

Policy sees immutable PublicObservation only. HiddenState belongs to the
environment/evaluator. The schema reference keeps a fixed 2*N scalar state;
it is a hand-designed sufficient statistic, never a learned TAC-OSM result.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
import copy
import hashlib
import json
import random


@dataclass(frozen=True)
class PublicObservation:
    tick: int
    entity: int
    kind: str
    value: int


@dataclass(frozen=True)
class Action:
    kind: str
    entity: int
    value: int = 0
    delay: int = 1

    def __post_init__(self):
        if self.kind not in ("inspect", "service", "wait"):
            raise ValueError("invalid action")
        if self.value not in (0, 1) or self.delay < 1:
            raise ValueError("binary value and positive delay required")


@dataclass(frozen=True)
class Receipt:
    observation: PublicObservation
    accepted: bool
    reason: str


@dataclass(frozen=True)
class Intervention:
    due: int
    entity: int
    generation: int
    value: int


class MaintenanceWorld:
    """Wear/failure state plus generation-guarded delayed service.

    A competing reset increments generation and invalidates an older pending
    service. External changes precede due service at the same tick. Inspection
    reveals only the chosen entity; waiting emits no state. No post-action
    receipt is available before the corresponding prediction is committed.
    """

    def __init__(self, bits: tuple[int, ...]):
        if not bits or any(x not in (0, 1) for x in bits):
            raise ValueError("nonempty binary initial state required")
        self._bits = list(bits)
        self._generation = [0] * len(bits)
        self._pending: list[Intervention] = []
        self.tick = 0

    def _entity(self, entity: int):
        if not 0 <= entity < len(self._bits):
            raise ValueError("entity outside world")

    def change(self, entity: int, value: int) -> PublicObservation:
        self._entity(entity)
        if value not in (0, 1):
            raise ValueError("binary change required")
        self._generation[entity] += 1
        self._bits[entity] = value
        return PublicObservation(self.tick, entity, "change", value)

    def act(self, action: Action) -> PublicObservation:
        self._entity(action.entity)
        if action.kind == "inspect":
            return PublicObservation(self.tick, action.entity, "inspect", self._bits[action.entity])
        if action.kind == "service":
            self._pending.append(Intervention(self.tick + action.delay, action.entity,
                                             self._generation[action.entity], action.value))
            return PublicObservation(self.tick, action.entity, "scheduled", action.value)
        return PublicObservation(self.tick, action.entity, "wait", -1)

    def advance(self, changes: tuple[tuple[int, int], ...] = ()) -> tuple[PublicObservation, ...]:
        self.tick += 1
        observations = [self.change(e, v) for e, v in changes]
        for item in self._pending:
            if item.due == self.tick and item.generation == self._generation[item.entity]:
                self._bits[item.entity] = item.value
                # Internal execution does not emit a hidden outcome to policy.
        self._pending = [p for p in self._pending if p.due > self.tick]
        return tuple(observations)

    def inspect_receipt(self, entity: int, claimed_value: int) -> Receipt:
        """Post-action verifier: compares the claim only with a fresh inspection."""
        if claimed_value not in (0, 1):
            raise ValueError("binary prediction required")
        obs = self.act(Action("inspect", entity))
        accepted = obs.value == claimed_value
        return Receipt(obs, accepted, "matches_observation" if accepted else "contradicts_observation")


class SchemaReference:
    """Fixed schema-indexed sufficient statistic, no historical event cache."""

    def __init__(self, entities: int):
        if entities < 1 or entities * 2 > 768:
            raise ValueError("schema exceeds scalar budget")
        self.bits = [-1] * entities
        self.generation = [0] * entities

    @property
    def state_scalars(self) -> int:
        return len(self.bits) + len(self.generation)

    def observe(self, obs: PublicObservation):
        if obs.kind in ("inspect", "change"):
            self.bits[obs.entity] = obs.value
        if obs.kind == "change":
            self.generation[obs.entity] += 1

    def predict(self, entity: int, action: Action) -> int:
        if self.bits[entity] not in (0, 1):
            raise ValueError("unobserved state")
        # The benchmark query asks about the selected intervention at its
        # maturity time, with an explicit no-future-exogenous-change horizon.
        return action.value if action.kind == "service" else self.bits[entity]


@dataclass(frozen=True)
class Case:
    observations: tuple[PublicObservation, ...]
    query_entity: int
    candidate: Action
    expected: int  # evaluator-only: never passed into reference.observe/predict
    fingerprint: str
    service_expected: int


def make_pair(seed: int, history: int, delay: int, entities: int = 8,
              *, domain: str = "audit", role: str = "retention") -> tuple[Case, Case]:
    if history < entities or entities < 2 or delay < 1:
        raise ValueError("invalid task dimensions")
    if role not in ("retention", "acquisition", "cancelled_service"):
        raise ValueError("unknown role")
    rng = random.Random(f"dev016:{domain}:{role}:{seed}:{history}:{delay}:{entities}")
    target = rng.randrange(entities)
    initial = [rng.randrange(2) for _ in range(entities)]
    others = [e for e in range(entities) if e != target]
    # The target is first observed early, then only competitors are changed.
    order = [target] + others
    schedule = [(rng.choice(others), rng.randrange(2)) for _ in range(history-entities)]
    pair = []
    for bit in (0, 1):
        bits = initial.copy()
        bits[target] = bit
        world = MaintenanceWorld(tuple(bits))
        observations = [world.act(Action("inspect", e)) for e in order]
        if role == "cancelled_service":
            observations.append(world.act(Action("service", target, 1, 2)))
            observations.extend(world.advance(((target, bit),)))
            observations.extend(world.advance())
        for e, v in schedule:
            observations.extend(world.advance(((e, v),)))
        # A delayed destructive service followed by a competing reset is
        # included in every history, so stale pending writes must be rejected.
        competitor = others[0]
        observations.append(world.act(Action("service", competitor, 1, 2)))
        observations.extend(world.advance(((competitor, 0),)))
        observations.extend(world.advance())
        observations.append(world.act(Action("inspect", competitor)))
        if role == "acquisition":
            observations.extend(world.advance(((target, bit),)))
        # Binary decision: predict target at horizon if we wait. A selected
        # service is evaluated separately as a counterfactual diagnostic.
        candidate = Action("wait", target, 0, delay)
        intervened = copy.deepcopy(world)
        intervened.act(Action("service", target, 0, delay))
        for _ in range(delay):
            world.advance()
            intervened.advance()
        expected = world.act(Action("inspect", target)).value
        service_expected = intervened.act(Action("inspect", target)).value
        payload = {"observations": [asdict(x) for x in observations],
                   "query_entity": target, "candidate": asdict(candidate)}
        fp = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
        pair.append(Case(tuple(observations), target, candidate, expected, fp, service_expected))
    return tuple(pair)
