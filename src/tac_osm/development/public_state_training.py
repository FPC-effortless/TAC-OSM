"""Public-only adapter and supervised objectives for the DEV-046 vertical slice."""
from __future__ import annotations
import copy

import torch

from .delayed_maintenance import Action
from .frozen_action_heads import action_head_loss
from .schema_maintenance import future_bits

KINDS = ("inspect", "change", "scheduled", "wait")


def public_tensors(streams):
    """Identity-free event content; IDs are routing addresses only, never features."""
    if not streams or not streams[0] or len({len(s) for s in streams}) != 1:
        raise ValueError("nonempty equal-length public histories required")
    features, entities, ticks = [], [], []
    for stream in streams:
        fs = []
        for o in stream:
            f = [0.] * 16
            f[8 + KINDS.index(o.kind)] = 1.
            f[12] = o.tick / 2048
            f[13] = max(0, o.due) / 2048
            if o.value in (0, 1):
                f[14 + o.value] = 1.
            fs.append(f)
        features.append(fs)
        entities.append([o.entity for o in stream])
        ticks.append([o.tick for o in stream])
    return (torch.tensor(features, dtype=torch.float32),
            torch.tensor(entities, dtype=torch.long), torch.tensor(ticks, dtype=torch.float32))


def prediction_from_state(model, state, horizons):
    return torch.stack(model.world_predictor(model.memory.read(state), horizons), -1)


def outcome_tensor(cases):
    return torch.tensor([list(zip(c.wait_targets, c.service_targets)) for c in cases], dtype=torch.float32)


def objective(probabilities, outcomes, kind):
    return action_head_loss(torch.logit(probabilities.clamp(1e-6, 1-1e-6)), outcomes, kind)


def student_rollout(model, state, cases, probabilities):
    """One student-encountered state: chosen services and one uncertainty inspection.

    Simulator labels remain outside all model calls. Rollout cost is training
    instrumentation, not a claim that inspections are free at evaluation.
    """
    world_copies = [copy.deepcopy(c.world) for c in cases]
    actions = 1+4*probabilities[..., 1] < 4*probabilities[..., 0]
    next_state = state
    service_count = 0
    for entity in range(probabilities.shape[1]):
        accepted = actions[:,entity].detach()
        receipts = [(world.act(Action("service" if bool(accepted[b]) else "wait", entity, 0, c.horizon)),)
                    for b,(world,c) in enumerate(zip(world_copies,cases))]
        public, ids, ticks = public_tensors(receipts)
        next_state = model.verified_update(next_state,public[:,0],ids[:,0],ticks[:,0],accepted)
        service_count += int(accepted.sum())
    inspect_entity = (probabilities[...,0]-.5).abs().argmin(-1).tolist()
    receipts = [(world.act(Action("inspect", entity)),) for world,entity in zip(world_copies,inspect_entity)]
    public, ids, ticks = public_tensors(receipts)
    next_state = model.verified_update(next_state,public[:,0],ids[:,0],ticks[:,0],
                                       torch.ones(len(cases),dtype=torch.bool))
    y = torch.tensor([list(zip(future_bits(w,c.horizon),future_bits(w,c.horizon,range(8))))
                      for w,c in zip(world_copies,cases)],dtype=torch.float32)
    return next_state,y,dict(services=service_count,inspections=len(cases))
