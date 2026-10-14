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
    due: int = -1


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
            return PublicObservation(self.tick, action.entity, "scheduled", action.value,
                                     self.tick + action.delay)
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
    """Bounded reference for schedules with one pending service per entity."""

    def __init__(self, entities: int):
        if entities < 1 or entities * 5 + 1 > 768:
            raise ValueError("schema exceeds scalar budget")
        self.bits = [-1] * entities
        self.generation = [0] * entities
        self.pending_due = [-1] * entities
        self.pending_value = [-1] * entities
        self.pending_generation = [-1] * entities
        self.tick = 0

    @property
    def state_scalars(self) -> int:
        return 5 * len(self.bits) + 1

    def _mature(self, tick: int):
        for e in range(len(self.bits)):
            if 0 <= self.pending_due[e] <= tick:
                if self.pending_generation[e] == self.generation[e]:
                    self.bits[e] = self.pending_value[e]
                self.pending_due[e] = -1
        self.tick = tick

    def observe(self, obs: PublicObservation):
        if obs.tick < self.tick:
            raise ValueError("observations must be chronological")
        if obs.tick > self.tick:
            self._mature(obs.tick - 1)
        if obs.kind in ("inspect", "change"):
            self.bits[obs.entity] = obs.value
        if obs.kind == "change":
            self.generation[obs.entity] += 1
        self._mature(obs.tick)
        if obs.kind == "scheduled":
            if self.pending_due[obs.entity] >= 0:
                raise ValueError("reference requires one pending service per entity")
            self.pending_due[obs.entity] = obs.due
            self.pending_value[obs.entity] = obs.value
            self.pending_generation[obs.entity] = self.generation[obs.entity]

    def predict(self, entity: int, action: Action) -> int:
        if self.bits[entity] not in (0, 1):
            raise ValueError("unobserved state")
        # The benchmark query asks about the selected intervention at its
        # maturity time, with an explicit no-future-exogenous-change horizon.
        future = copy.deepcopy(self)
        future._mature(self.tick + action.delay)
        return action.value if action.kind == "service" else future.bits[entity]


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
    if role not in ("retention", "acquisition", "cancelled_service", "pending_cancel"):
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
        bits[target] = 1 if role == "pending_cancel" else bit
        world = MaintenanceWorld(tuple(bits))
        observations = [world.act(Action("inspect", e)) for e in order]
        if role == "pending_cancel":
            # Both partners visibly start failed and receive the same service.
            # Only a historical reset determines whether it can mature. A
            # latest-value lookup cannot distinguish future outcomes here.
            observations.append(world.act(Action("service", target, 0,
                                                       history - entities + 3 + delay)))
            changed_entity = target if bit else others[-1]
            observations.extend(world.advance(((changed_entity, 1),)))
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
        observations.append(world.act(Action("wait", target)))
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
