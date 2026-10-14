"""DEV-017 exact-length histories, unresolved interventions and physical cost.

Everything in Episode beyond observations/tick/horizon is evaluator-only.
Learned policies never receive hidden worlds, generations, targets or roles.
"""
from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
import hashlib
import json
import random

from .delayed_maintenance import Action, MaintenanceWorld, PublicObservation

ROLES = ("retention", "acquisition", "pending_cancel", "reissued_cancel")
DEADLINES = ("before", "at", "after")
FAILURE_COST = 4.0
SERVICE_COST = 1.0
INSPECTION_COST = .1


@dataclass(frozen=True)
class Episode:
    observations: tuple[PublicObservation, ...]
    tick: int
    horizon: int
    target_entity: int
    fingerprint: str
    current_targets: tuple[int, ...]
    wait_targets: tuple[int, ...]
    service_targets: tuple[int, ...]
    world: MaintenanceWorld


def future_bits(world: MaintenanceWorld, horizon: int, service_entities=()) -> tuple[int, ...]:
    if horizon < 1:
        raise ValueError("positive forecast horizon required")
    clone = copy.deepcopy(world)
    for entity in service_entities:
        clone.act(Action("service", entity, 0, horizon))
    for _ in range(horizon):
        clone.advance()
    return tuple(clone.act(Action("inspect", e)).value for e in range(8))


def make_pair(seed: int, history: int, due_delay: int, role: str,
              deadline: str, *, domain: str = "eval") -> tuple[Episode, Episode]:
    if history < 32 or due_delay < 4 or role not in ROLES or deadline not in DEADLINES:
        raise ValueError("invalid DEV-017 condition")
    rng = random.Random(f"dev017-v1:{domain}:{seed}:{history}:{due_delay}:{role}:{deadline}")
    target = rng.randrange(8)
    others = [e for e in range(8) if e != target]
    order = list(range(8))
    rng.shuffle(order)
    initial = [rng.randrange(2) for _ in range(8)]
    distraction = [(rng.choice(others), rng.randrange(2)) for _ in range(history)]
    horizon = {"before": due_delay // 2, "at": due_delay,
               "after": due_delay + due_delay // 2}[deadline]
    episodes = []
    for partner in (0, 1):
        bits = initial.copy()
        bits[target] = 1 if "cancel" in role else partner
        world = MaintenanceWorld(tuple(bits))
        stream = []
        for tick in range(1, history + 1):
            world.advance()
            if tick <= 8:
                obs = world.act(Action("inspect", order[tick - 1]))
            elif role in ("pending_cancel", "reissued_cancel") and tick == 9:
                obs = world.act(Action("service", target, 0, history + due_delay - tick))
            elif role == "pending_cancel" and tick == 10:
                obs = world.change(target if partner else others[0], 1)
            elif role == "reissued_cancel" and tick == 10:
                obs = world.change(target, 1)  # invalidate first generation
            elif role == "reissued_cancel" and tick == 11:
                obs = world.act(Action("service", target, 0, history + due_delay - tick))
            elif role == "reissued_cancel" and tick == 12:
                obs = world.change(target if partner else others[0], 1)
            elif role == "acquisition" and tick == history:
                obs = world.change(target, partner)
            else:
                entity, value = distraction[tick - 1]
                obs = world.change(entity, value)
            stream.append(obs)
        # Current labels are supervisor-only. Inference may acquire one of
        # these bits only through a chosen inspect action and charged receipt.
        current = tuple(world.act(Action("inspect", e)).value for e in range(8))
        waiting = future_bits(world, horizon)
        serviced = future_bits(world, horizon, range(8))
        payload = {"observations": [asdict(o) for o in stream],
                   "tick": history, "horizon": horizon}
        fingerprint = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
        episodes.append(Episode(tuple(stream), history, horizon, target, fingerprint,
                                current, waiting, serviced, world))
    if deadline != "before" and episodes[0].wait_targets[target] == episodes[1].wait_targets[target]:
        raise AssertionError("mature target counterfactuals must have opposite outcomes")
    return tuple(episodes)


def execute(world: MaintenanceWorld, horizon: int, services: tuple[int, ...],
            inspections: int) -> dict:
    if len(set(services)) != len(services) or not 0 <= inspections <= 2:
        raise ValueError("invalid intervention/inspection budget")
    wait = future_bits(world, horizon)
    outcome = future_bits(world, horizon, services)
    failures = sum(outcome)
    unnecessary = sum(wait[e] == 0 for e in services)
    cost = FAILURE_COST * failures + SERVICE_COST * len(services) + INSPECTION_COST * inspections
    return {"outcomes": list(outcome), "wait_outcomes": list(wait), "failures": failures,
            "services": list(services), "unnecessary_services": unnecessary,
            "inspections": inspections, "total_cost": cost, "cost_per_entity": cost / 8}


def exact_public_targets(stream: tuple[PublicObservation, ...], tick: int,
                         horizon: int) -> tuple[int, ...]:
    """41-scalar evaluator reference; NEVER a learned policy or model feature.

    Five arrays (current bit/generation/eligible due/value/generation) and tick.
    A generation change makes older queued interventions causally irrelevant.
    This reference only validates the restricted generator's solvability.
    """
    bits, generation = [-1] * 8, [0] * 8
    due, value, scheduled_generation = [-1] * 8, [-1] * 8, [-1] * 8
    for obs in stream:
        e = obs.entity
        if obs.kind in ("change", "inspect"):
            bits[e] = obs.value
        if obs.kind == "change":
            generation[e] += 1
            due[e] = -1
        if obs.kind == "scheduled":
            if due[e] >= 0:
                raise AssertionError("more than one causally eligible intervention")
            due[e], value[e], scheduled_generation[e] = obs.due, obs.value, generation[e]
    for e in range(8):
        if tick < due[e] <= tick + horizon and scheduled_generation[e] == generation[e]:
            bits[e] = value[e]
    if any(v not in (0, 1) for v in bits):
        raise AssertionError("reference has unobserved entity")
    return tuple(bits)
