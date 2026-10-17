"""DEV-042 paired public histories: causal schedule/reset order versus identical last events.

No model target is disclosed to the policy. Evaluator labels and hidden world
are only constructed for verification. This changes the generator distribution;
it does not modify the older DEV-041 benchmark or its results.
"""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import random

from .delayed_maintenance import Action, MaintenanceWorld
from .schema_maintenance import Episode, future_bits, exact_public_targets


def last_observations(stream):
    latest = {}
    for obs in stream:
        latest[obs.entity] = obs
    return tuple(latest[e] for e in range(8))


def make_order_pair(seed: int, history: int, due_delay: int, deadline: str,
                    *, domain: str = "dev042-evaluation-v1"):
    if history < 32 or due_delay < 4 or deadline not in ("before", "at", "after"):
        raise ValueError("invalid paired causal-history condition")
    rng = random.Random(f"dev042:v1:{domain}:{seed}:{history}:{due_delay}:{deadline}")
    target = rng.randrange(8)
    order = list(range(8))
    rng.shuffle(order)
    others = [e for e in range(8) if e != target]
    initial = [rng.randrange(2) for _ in range(8)]
    initial[target] = 1
    distraction = [(rng.choice(others), rng.randrange(2)) for _ in range(history)]
    horizon = {"before":due_delay//2, "at":due_delay,
               "after":due_delay + due_delay//2}[deadline]
    pair = []
    for partner in (0, 1):
        world = MaintenanceWorld(tuple(initial))
        stream = []
        for t in range(1, history+1):
            world.advance()
            if t <= 8:
                obs = world.act(Action("inspect", order[t-1]))
            elif t == 9:
                if partner == 0:
                    obs = world.act(Action("service", target, 0, history+due_delay-t))
                else:
                    obs = world.change(target, 1)
            elif t == 10:
                if partner == 0:
                    obs = world.change(target, 1)
                else:
                    obs = world.act(Action("service", target, 0, history+due_delay-t))
            elif t == history:
                obs = world.act(Action("inspect", target))
            else:
                e, value = distraction[t-1]
                obs = world.change(e, value)
            stream.append(obs)
        current = tuple(world.act(Action("inspect",e)).value for e in range(8))
        waiting = future_bits(world, horizon)
        serviced = future_bits(world,horizon,range(8))
        payload = {"observations":[asdict(o) for o in stream],
                   "tick":history, "horizon":horizon}
        fingerprint = hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()
        episode=Episode(tuple(stream),history,horizon,target,fingerprint,
                        current,waiting,serviced,world)
        if exact_public_targets(episode.observations,history,horizon)!=waiting:
            raise AssertionError("independent public-history reference mismatch")
        pair.append(episode)
    a,b = pair
    if a.fingerprint==b.fingerprint or (a.tick,a.horizon,a.target_entity)!=(b.tick,b.horizon,b.target_entity):
        raise AssertionError("different histories, identical public queries required")
    if last_observations(a.observations)!=last_observations(b.observations):
        raise AssertionError("last event shortcut must be indistinguishable")
    if a.current_targets!=b.current_targets or a.current_targets[target]!=1:
        raise AssertionError("target current state must be identical")
    if deadline=="before":
        if a.wait_targets != b.wait_targets:
            raise AssertionError("future should not diverge before scheduled due")
    else:
        if (a.wait_targets[target],b.wait_targets[target])!=(1,0):
            raise AssertionError("causal ordering must reverse future target")
        if any(a.wait_targets[e]!=b.wait_targets[e] for e in others):
            raise AssertionError("no other entity causal differences allowed")
    if a.service_targets!=b.service_targets or a.service_targets[target]!=0:
        raise AssertionError("same explicit service intervention must recover target")
    return tuple(pair)
