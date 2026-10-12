"""Paired retention AND fresh-acquisition tasks without visible role labels."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import random

import torch


@dataclass(frozen=True)
class Event:
    address: tuple[float,...]
    value: int


@dataclass(frozen=True)
class Episode:
    history: tuple[Event,...]
    writes: tuple[Event,...]
    query: tuple[float,...]
    current_bit: int
    target: int                    # evaluator only
    target_stream_slot: int        # evaluator only
    task: str                      # evaluator only


@dataclass(frozen=True)
class Pair:
    left: Episode
    right: Episode


def visible(ep: Episode) -> dict:
    return {'addresses':tuple(e.address for e in ep.history+ep.writes),
            'values':tuple(e.value for e in ep.history+ep.writes),
            'query':ep.query,'current_bit':ep.current_bit}


def fingerprint(ep: Episode) -> str:
    return hashlib.sha256(json.dumps(visible(ep),sort_keys=True).encode()).hexdigest()


def seed_for(namespace: str,*parts: object) -> int:
    data = json.dumps(['RETENTION-PROTECTION-DEV-005',namespace,*parts],separators=(',',':'))
    return int.from_bytes(hashlib.sha256(data.encode()).digest()[:8],'big')


def make_pair(seed: int,h: int,d: int,fraction: float,noise: float,task: str) -> Pair:
    if h not in (32,128,512) or d not in (4,16,32) or fraction not in (0.0,1.0) or noise not in (.08,.25):
        raise ValueError('unregistered development condition')
    if task not in ('retention','acquisition'):
        raise ValueError('unknown task')
    rng = random.Random(seed)
    def gaussian():
        return tuple(rng.gauss(0,1) for _ in range(12))
    def perturb(vector,scale):
        return tuple(round(x+rng.gauss(0,scale),8) for x in vector)
    latent_history = [gaussian() for _ in range(h)]
    anchor = rng.randrange(h)
    latent_writes = [perturb(latent_history[anchor],.12) if fraction else gaussian() for _ in range(d)]
    latent = latent_history+latent_writes
    target_slot = anchor if task=='retention' else h+rng.randrange(d)
    payload = [rng.randrange(2) for _ in latent]
    addresses = [perturb(z,noise) for z in latent]
    query = perturb(latent[target_slot],noise)
    current = rng.randrange(2)
    events = tuple(Event(a,v) for a,v in zip(addresses,payload))
    flipped = tuple(Event(e.address,1-e.value if i==target_slot else e.value) for i,e in enumerate(events))
    left = Episode(events[:h],events[h:],query,current,payload[target_slot]^current,target_slot,task)
    right = Episode(flipped[:h],flipped[h:],query,current,(1-payload[target_slot])^current,target_slot,task)
    pair = Pair(left,right)
    validate_pair(pair,h,d)
    return pair


def validate_pair(pair: Pair,h: int,d: int):
    a,b = pair.left,pair.right
    if len(a.history)!=h or len(a.writes)!=d or len(b.history)!=h or len(b.writes)!=d:
        raise AssertionError('pair shape')
    if a.query!=b.query or a.current_bit!=b.current_bit or a.target==b.target or a.task!=b.task:
        raise AssertionError('counterfactual present/labels')
    events,other = a.history+a.writes,b.history+b.writes
    if sum(x!=y for x,y in zip(events,other))!=1 or any(x.address!=y.address for x,y in zip(events,other)):
        raise AssertionError('non-target difference')
    if a.target_stream_slot!=b.target_stream_slot:
        raise AssertionError('oracle identity differs')
    if a.query in tuple(e.address for e in events):
        raise AssertionError('exact-key shortcut')
    for ep in (a,b):
        index = ep.target_stream_slot
        if not 0<=index<h+d or ep.target!=(ep.history+ep.writes)[index].value^ep.current_bit:
            raise AssertionError('incorrect target')
        if (index<h)!=(ep.task=='retention'):
            raise AssertionError('wrong task boundary')
        if set(visible(ep))!={'addresses','values','query','current_bit'}:
            raise AssertionError('oracle input leakage')


def tensors(episodes: list[Episode]):
    views = [visible(e) for e in episodes]
    return (torch.tensor([v['addresses'] for v in views],dtype=torch.float32),
            torch.tensor([v['values'] for v in views],dtype=torch.long),
            torch.tensor([v['query'] for v in views],dtype=torch.float32),
            torch.tensor([v['current_bit'] for v in views],dtype=torch.long))


def euclidean(ep: Episode) -> tuple[int,int]:
    v = visible(ep)
    distances = [math.fsum((x-y)**2 for x,y in zip(v['query'],a)) for a in v['addresses']]
    index = min(range(len(distances)),key=lambda i:distances[i])
    return v['values'][index]^v['current_bit'],index
