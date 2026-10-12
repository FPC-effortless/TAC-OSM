"""Development-only action-conditioned latent targets and a TAC-OSM prediction head.

No oracle metadata crosses into inference; the artificial latent target is
created exclusively in the training/evaluator boundary.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from tac_osm.development.conditional_compute import selected_weights
from tac_osm.development.protected_memory import ProtectionModel
from tac_osm.development.protection_benchmark import Episode, make_pair, seed_for, tensors


ARMS = ("task_only", "future_only", "joint", "joint_shuffled")


def future_target(past: Tensor, present: Tensor, action: Tensor) -> Tensor:
    """Evaluator-only, signed action-controlled transition. Inputs all [B] bits."""
    if not (past.shape == present.shape == action.shape and past.ndim == 1):
        raise ValueError("future target needs matched one-dimensional bit vectors")
    for bit in (past, present, action):
        if not bool(((bit == 0) | (bit == 1)).all()):
            raise ValueError("expected binary inputs")
    m, c, a = (2 * past.float() - 1), (2 * present.float() - 1), (2 * action.float() - 1)
    return torch.stack((0.7*m+0.4*a*c, 0.8*c+0.4*a*m, 0.6*a*m+0.25*c), dim=-1)


@dataclass(frozen=True)
class WorldBatch:
    addresses: Tensor
    values: Tensor
    query: Tensor
    current: Tensor
    action: Tensor
    answer: Tensor       # evaluator only
    next_latent: Tensor  # evaluator only
    tasks: tuple[str, ...]

    def inference(self) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """The *only* fields allowed to enter the model forward method."""
        return self.addresses, self.values, self.query, self.current, self.action


def make_world_batch(seed: int, step: int, *, smoke: bool = False) -> WorldBatch:
    # Training uses full opposite-past pairs, same action and present per pair.
    h = (32,) if smoke else (32,128)
    d = (4,) if smoke else (4,16)
    hs = h[step % len(h)]
    ds = d[(step//2) % len(d)]
    fraction = (0.0,1.0)[(step//4)%2]
    noise = (.08,.25)[(step//8)%2]
    episodes: list[Episode] = []
    actions: list[int] = []
    tasks: list[str] = []
    for j in range(4):
        role = ('retention','acquisition')[j%2]
        pair = make_pair(seed_for('predworld-train',seed,step,j),hs,ds,fraction,noise,role)
        episodes.extend((pair.left,pair.right))
        actions.extend(((step+j)%2,)*2)
        tasks.extend((role,)*2)
    addresses,values,query,current = tensors(episodes)
    answer = torch.tensor([e.target for e in episodes],dtype=torch.long)
    past = answer ^ current   # Evaluator-only historical bit. Never a model input.
    action = torch.tensor(actions,dtype=torch.long)
    return WorldBatch(addresses,values,query,current,action,answer,
                      future_target(past,current,action),tuple(tasks))


class PredictiveWorldModel(ProtectionModel):
    """Preserves the DEV-005 memory and operator path; adds future prediction."""
    def __init__(self):
        super().__init__(mode='always_update')
        d = self.config.dim
        self.world = nn.Sequential(nn.Linear(2*d+2,d),nn.GELU(),nn.Linear(d,3))

    def predict_both(self, state: Tensor, address: Tensor,
                     current: Tensor, action: Tensor) -> tuple[Tensor,Tensor]:
        if address.ndim != 2 or action.shape != (address.shape[0],):
            raise ValueError("query and action shapes are incompatible")
        if not bool(((action==0)|(action==1)).all()):
            raise ValueError("action must be binary")
        onehot = F.one_hot(current.long(),2).to(address.dtype)
        query = self.query(torch.cat((address,onehot),dim=-1))
        reads = torch.einsum('bevk,bek->bev',state,self.memory.keys(address))
        views = self.read_projection(reads)
        scores = self.read_selector(torch.cat((views,query[:,None].expand_as(views)),-1)).squeeze(-1)
        indices,weights = selected_weights(scores,self.config.read_top_k,self.training)
        selected = views.gather(1,indices[...,None].expand(-1,-1,views.shape[-1]))
        routed = (selected*weights.gather(1,indices)[...,None]).sum(1)
        computation,_ = self.compute(routed,query)
        features = torch.cat((computation,query),dim=-1)
        action_onehot = F.one_hot(action.long(),2).to(address.dtype)
        return self.answer(features), self.world(torch.cat((features,action_onehot),dim=-1))


def arm_loss(arm: str, task_logits: Tensor, future: Tensor,
             labels: Tensor, next_latent: Tensor) -> tuple[Tensor,Tensor,Tensor]:
    if arm not in ARMS:
        raise ValueError(f"unknown arm: {arm}")
    ce = F.cross_entropy(task_logits,labels)
    mse = F.mse_loss(future,next_latent)
    if arm=='task_only':
        return ce,ce,mse.detach()
    if arm=='future_only':
        return mse,ce.detach(),mse
    if arm=='joint':
        return ce+mse,ce,mse
    # Destroy mapping to the correct sample/past/action while preserving
    # target distribution. Rotate complete pairs, never within a pair.
    corrupt = next_latent.roll(shifts=2,dims=0)
    return ce+F.mse_loss(future,corrupt),ce,mse


def smoke_train(seed: int = 43, steps: int = 2) -> dict:
    """Fast objective/gradient check; no capability result or holdout."""
    if not 1 <= steps <= 2:
        raise ValueError("smoke budget limited to two steps")
    torch.set_num_threads(min(2,torch.get_num_threads()))
    torch.manual_seed(seed)
    source = PredictiveWorldModel()
    initial = {k:v.detach().clone() for k,v in source.state_dict().items()}
    outcomes = {}
    for arm in ARMS:
        model = PredictiveWorldModel()
        model.load_state_dict(initial)
        opt = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),lr=.003)
        losses=[]
        for step in range(steps):
            batch = make_world_batch(seed,step,smoke=True)
            model.train()
            state = model.encode(batch.addresses,batch.values,trace=False)
            task, world = model.predict_both(state,batch.query,batch.current,batch.action)
            loss,_,_ = arm_loss(arm,task,world,batch.answer,batch.next_latent)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            # A world-only future objective MUST have gradient to a memory writer.
            if not (model.memory.write.weight.grad is not None and
                    model.memory.write.weight.grad.abs().sum().item() > 0):
                raise AssertionError(f'{arm}: no gradient to recurrent write')
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
            opt.step()
            losses.append(float(loss.detach()))
        model.eval()
        with torch.no_grad():
            batch = make_world_batch(seed+100,0,smoke=True)
            state = model.encode(batch.addresses,batch.values,trace=False)
            task, world = model.predict_both(state,batch.query,batch.current,batch.action)
            outcomes[arm] = {'smoke_losses':losses,
                             'smoke_task_accuracy':float((task.argmax(-1)==batch.answer).float().mean()),
                             'smoke_latent_mse':float(F.mse_loss(world,batch.next_latent))}
    return outcomes
