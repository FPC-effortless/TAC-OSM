"""DEV-047 public-ID maintenance cohorts, variable entities and queued interventions.

World/targets are evaluator-only. Catalogs and observations are the complete
deployment boundary; catalog registration is supplied, not learned discovery.
"""
from __future__ import annotations
import copy
from dataclasses import dataclass, asdict, replace
import hashlib
import json
import random

from .delayed_maintenance import Action, MaintenanceWorld, PublicObservation


@dataclass(frozen=True)
class ScalableEpisode:
    observations: tuple[PublicObservation, ...]
    entity_ids: tuple[int, ...]
    tick: int
    horizon: int
    target_entity: int
    fingerprint: str
    wait_targets: tuple[int, ...]
    service_targets: tuple[int, ...]
    world: MaintenanceWorld


def future(world, entities, horizon, services=()):
    clone = copy.deepcopy(world)
    for entity in services:
        clone.act(Action("service",entity,0,horizon))
    for _ in range(horizon):
        clone.advance()
    return tuple(clone.act(Action("inspect",e)).value for e in range(entities))


def make_pair(seed, history, delay, role, deadline, *, entities=8, domain="dev047", multiple_pending=False, unseen_ids=False):
    if entities<2 or history<entities+6 or delay<4:
        raise ValueError("insufficient initialization/history or delay")
    if role not in ("retention","acquisition","pending_cancel","reissued_cancel") or deadline not in ("before","at","after"):
        raise ValueError("unknown condition")
    rng = random.Random(f"dev047-v1:{seed}:{history}:{delay}:{role}:{deadline}:{entities}:{domain}:{multiple_pending}")
    target = rng.randrange(entities)
    others = [e for e in range(entities) if e!=target]
    initial = [rng.randrange(2) for _ in range(entities)]
    order = rng.sample(range(entities),entities)
    catalog = tuple(rng.sample(range(100000,100000+entities*100),entities)) if unseen_ids else tuple(range(entities))
    horizon = {"before":delay//2,"at":delay,"after":delay+delay//2}[deadline]
    distractions = [(rng.choice(others),rng.randrange(2)) for _ in range(history)]
    result = []
    for partner in (0,1):
        bits = initial.copy()
        bits[target] = 1 if "cancel" in role else partner
        world = MaintenanceWorld(tuple(bits))
        observations = []
        for tick in range(1,history+1):
            world.advance()
            if tick<=entities:
                obs = world.act(Action("inspect",order[tick-1]))
            elif "cancel" in role and tick==entities+1:
                obs = world.act(Action("service",target,0,history+delay-tick))
            elif "cancel" in role and tick==entities+2:
                obs = world.change(target if (role=="reissued_cancel" or partner) else others[0],1)
            elif role=="reissued_cancel" and tick==entities+3:
                obs = world.act(Action("service",target,0,history+delay-tick))
            elif role=="reissued_cancel" and tick==entities+4:
                obs = world.change(target if partner else others[0],1)
            elif multiple_pending and tick==entities+5:
                obs = world.act(Action("service",target,1,history+delay-1-tick))
            elif role=="acquisition" and tick==history:
                obs = world.change(target,partner)
            else:
                obs = world.change(*distractions[tick-1])
            observations.append(replace(obs,entity=catalog[obs.entity]))
        payload = dict(observations=[asdict(o) for o in observations],catalog=catalog,tick=history,horizon=horizon)
        fp = hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()
        result.append(ScalableEpisode(tuple(observations),catalog,history,horizon,target,fp,
            future(world,entities,horizon),future(world,entities,horizon,range(entities)),world))
    return tuple(result)


def public_reference(observations, catalog, tick, horizon):
    """Independent event-queue reference using only public history, no world clone."""
    index = {key:i for i,key in enumerate(catalog)}
    if len(index)!=len(catalog):
        raise ValueError("unique public identity catalog required")
    bits = [-1]*len(catalog)
    generations = [0]*len(catalog)
    pending = []
    def mature(bound, inclusive):
        nonlocal pending
        due_now = [p for p in pending if p[0]<=bound if inclusive or p[0]<bound]
        for due,serial,e,g,value in sorted(due_now):
            if generations[e]==g:
                bits[e] = value
        pending = [p for p in pending if p not in due_now]
    previous = -1
    for serial,o in enumerate(observations):
        if o.tick<previous or o.tick>tick:
            raise ValueError("chronological public history required")
        mature(o.tick,False)
        e = index[o.entity]
        if o.kind=="change":
            generations[e] += 1
            bits[e] = o.value
        mature(o.tick,True)
        if o.kind=="inspect":
            bits[e] = o.value
        elif o.kind=="scheduled":
            pending.append((o.due,serial,e,generations[e],o.value))
        previous = o.tick
    mature(tick+horizon,True)
    if any(x not in (0,1) for x in bits):
        raise ValueError("unobserved public entity")
    return tuple(bits)


def catalog_tensors(streams,catalogs):
    from .public_state_training import public_tensors
    if len(streams)!=len(catalogs):
        raise ValueError("one catalog per stream")
    canonical = []
    for stream,catalog in zip(streams,catalogs):
        index = {key:i for i,key in enumerate(catalog)}
        if len(index)!=len(catalog):
            raise ValueError("duplicate public identity")
        canonical.append(tuple(replace(o,entity=index[o.entity]) for o in stream))
    return public_tensors(canonical)


def execute(case, services, inspections=0):
    n = len(case.entity_ids)
    if len(set(services))!=len(services) or any(not 0<=e<n for e in services) or inspections<0:
        raise ValueError("invalid action set")
    outcome = future(case.world,n,case.horizon,services)
    failures = sum(outcome)
    unnecessary = sum(case.wait_targets[e]==0 for e in services)
    cost = 4*failures+len(services)+.1*inspections
    return dict(outcomes=list(outcome),services=list(services),failures=failures,
                unnecessary_services=unnecessary,inspections=inspections,total_cost=cost,cost_per_entity=cost/n)
