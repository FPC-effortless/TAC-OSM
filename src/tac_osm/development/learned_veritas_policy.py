"""DEV-051: learn public maintenance actions from teacher and student-encountered states.

The policy and its teacher see only stale public records and observable receipts.
Native hidden Veritas state is confined to the evaluator in run_episode().
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
import time

import torch
from torch import nn
import torch.nn.functional as F

ACTIONS=("inspect","isolate","service","done")
ACTION_ID={name:i for i,name in enumerate(ACTIONS)}
FEATURE_DIM=12


@dataclass
class PublicBelief:
    reported_fault:int
    known_fault:int|None=None
    attempts:int=0
    isolated:bool=False
    ever_inspected:bool=False
    last_action:str|None=None
    last_accepted:bool=False

    def features(self):
        return [
            float(self.reported_fault),float(self.known_fault is not None),
            float(self.known_fault or 0),self.attempts/2.,
            float(self.isolated),float(self.ever_inspected),
            float(self.attempts>0 and self.known_fault is None),
            float(self.last_action=="inspect"),float(self.last_action=="isolate"),
            float(self.last_action=="service"),float(self.last_accepted),1.,
        ]

    def legal(self):
        # DONE is always possible and not necessarily safe; this is NOT
        # a hand-coded teacher choice disguised as admissibility.
        return [True,not self.isolated,self.isolated and self.attempts<2,True]

    def observe(self,name,result):
        self.last_action=name
        self.last_accepted=bool(result.get("accepted",True))
        if name=="inspect":
            if "fault" not in result:raise ValueError("sensor receipt lacks public fault field")
            self.known_fault=int(result["fault"])
            self.ever_inspected=True
        elif name=="isolate" and self.last_accepted:
            self.isolated=True
        elif name=="service" and self.last_accepted:
            self.attempts+=1
            self.isolated=False
            self.known_fault=None
        if self.attempts>2:raise AssertionError("unsupported service attempt count")


def public_teacher(state):
    """Known explicit public-belief reference; never reads hidden simulator."""
    if state.known_fault is None:return ACTION_ID["inspect"]
    if state.known_fault==0:return ACTION_ID["done"]
    if state.isolated and state.attempts<2:return ACTION_ID["service"]
    if not state.isolated:return ACTION_ID["isolate"]
    return ACTION_ID["done"]


class PublicActionPolicy(nn.Module):
    def __init__(self,seed):
        super().__init__()
        torch.manual_seed(seed)
        self.net=nn.Sequential(nn.Linear(FEATURE_DIM,32),nn.Tanh(),
                               nn.Linear(32,32),nn.Tanh(),nn.Linear(32,len(ACTIONS)))

    def logits(self,x):
        if x.ndim!=2 or x.shape[1]!=FEATURE_DIM:
            raise ValueError("only explicitly constructed public belief features accepted")
        return self.net(x)


def choose_action(state,model=None,confidence_gate=None):
    teacher=public_teacher(state)
    if model is None:
        return teacher,True,1.
    x=torch.tensor([state.features()],dtype=torch.float32)
    with torch.no_grad():
        scores=model.logits(x)[0]
        valid=torch.tensor(state.legal(),dtype=torch.bool)
        scores=scores.masked_fill(~valid,-torch.inf)
        probs=scores.softmax(-1)
        arg=int(probs.argmax())
        confidence=float(probs[arg])
    fallback=(confidence_gate is not None and confidence<confidence_gate)
    return (teacher if fallback else arg),bool(fallback),confidence


def _native_runtime(episode):
    from investigation_world.operational.runtime import OperationalRuntime
    return OperationalRuntime(episode)


def rollout(episode,name,model=None,confidence_gate=None,collect=False):
    """Run a policy against Veritas, charging every call; teacher labels public-only."""
    from investigation_world.operational.models import EpisodeSubmission

    runtime=_native_runtime(episode)
    public=runtime.public_payload()
    if "oracle" in public or "initial_state" in json.dumps(public):
        raise AssertionError("privileged field crossed the policy boundary")
    actions=[];examples=[];fallbacks=0;decisions=0
    for record in public["records"]:
        entity=int(record["fields"]["entity"])
        belief=PublicBelief(reported_fault=int(record["fields"]["reported_fault"]))
        for _ in range(10):
            feature=belief.features()
            label=public_teacher(belief)
            action,used_fallback,confidence=choose_action(
                belief,model,confidence_gate)
            if action not in range(len(ACTIONS)) or not belief.legal()[action]:
                raise AssertionError("action violates explicit public feasibility constraints")
            if collect:
                examples.append(dict(x=feature,label=label,student_action=action,
                                     fallback=used_fallback))
            decisions+=1
            fallbacks+=int(used_fallback and model is not None)
            if action==ACTION_ID["done"]:
                actions.append(dict(action="done",entity=entity,
                         confidence=confidence,teacher_label=label))
                break
            kind=ACTIONS[action]
            parameters=dict(entity=entity)
            if kind=="service":parameters["attempt"]=belief.attempts+1
            receipt=runtime.act(kind,**parameters)
            if kind=="inspect":
                # Declared DEV-050 observable sensor boundary: native action is
                # charged, only *this* inspected entity's fault is returned.
                receipt["fault"]=runtime.state[f"e{entity}.fault"]
            result=dict(receipt)
            actions.append(dict(action=kind,parameters=parameters,
                result=result,confidence=confidence,teacher_label=label))
            belief.observe(kind,result)
        else:
            actions.append(dict(action="step_limit",entity=entity))
    state=runtime.state_snapshot()
    verified=runtime.submit(EpisodeSubmission(conclusion="DEV-051 public policy"))
    failures=sum(int(state[f"e{e}.fault"]) for e in range(2))
    return dict(policy=name,world=public["episode_id"],failures=failures,
                complete=int(verified.state_score==1.) if hasattr(verified,"state_score") else
                         int(verified.model_dump(mode="json")["state"]==1),
                total_cost=runtime.budget.spent/10+4*failures,
                blocked=sum(int(e.blocked) for e in runtime.events),
                tool_calls=len(runtime.events),sensing_calls=sum(e.action_name=="inspect"
                                    for e in runtime.events),
                fallbacks=fallbacks,decisions=decisions,
                actions=actions,
                verification=verified.model_dump(mode="json")),examples


def public_cohort(seeds,per_seed):
    from .veritas_public_challenge import make_episode
    for seed in seeds:
        for i in range(per_seed):
            yield seed,i,make_episode(seed,i)


def supervised_samples(worlds,model=None):
    features=[];target=[];states=0
    for _,_,episode in worlds:
        _,rows=rollout(episode,"public_teacher" if model is None else "student",
                       model=model,collect=True)
        for row in rows:
            features.append(row["x"])
            target.append(row["label"])
        states+=1
    return torch.tensor(features,dtype=torch.float32),torch.tensor(target,dtype=torch.long),states


def fit(model,x,y,updates,lr=.003,optimizer=None):
    opt=optimizer or torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=.0001)
    curve=[]
    for step in range(updates):
        opt.zero_grad(set_to_none=True)
        logits=model.logits(x)
        loss=F.cross_entropy(logits,y)
        if not torch.isfinite(loss):raise FloatingPointError("nonfinite policy loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
        opt.step()
        if step==0 or step==updates-1:
            curve.append(dict(step=step+1,loss=float(loss.detach())))
    return opt,curve


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def aggregate(results):
    if not results:raise ValueError("empty evaluation")
    entities=2*len(results)
    return dict(worlds=len(results),cost_per_entity=sum(r["total_cost"] for r in results)/entities,
                residual_faults=sum(r["failures"] for r in results),
                complete_worlds=sum(r["complete"] for r in results),
                blocked_actions=sum(r["blocked"] for r in results),
                mean_tool_calls=sum(r["tool_calls"] for r in results)/len(results),
                mean_sensing_calls=sum(r["sensing_calls"] for r in results)/len(results),
                fallback_fraction=sum(r["fallbacks"] for r in results)/
                    max(1,sum(r["decisions"] for r in results)))
