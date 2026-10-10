"""Exploratory learned address + bounded write-interference pilot for 002.

NOT a confirmatory experiment. Training receives only observable corrupted
query/history addresses and an oracle *supervision label*; inference receives
NO latent key, target slot, target value, condition identifier, or oracle rank.
The fixed-capacity associative cache is an explicit nonlearned stress baseline,
NOT a learned recurrent memory, and cannot establish TAC-OSM integration.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import random
from typing import Sequence

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from tac_osm.integration_002_benchmark import (
    DIMS, Episode, Pair, make_pair, visible_inputs,
)

TRAIN_SEEDS = (43, 59, 71)
EVAL_SEEDS = (131, 149, 167)
TRAIN_STEPS = 180
TRAIN_BATCH = 12
EVAL_PAIRS_PER_CELL = 5
TRAIN_NAMESPACE = 37_000_000
EVAL_NAMESPACE = 67_000_000
CACHE_BUCKETS = 64


class LearnedNoisyMetric(nn.Module):
    """Low-capacity learned Mahalanobis scorer; identity equals Euclidean."""

    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(DIMS, DIMS, bias=False)
        with torch.no_grad():
            self.projection.weight.copy_(torch.eye(DIMS))

    def forward(self, query: Tensor, candidates: Tensor) -> Tensor:
        if query.ndim != 2 or candidates.ndim != 3:
            raise ValueError("query [batch,dim], candidates [batch,count,dim]")
        if query.shape[0] != candidates.shape[0] or query.shape[-1] != DIMS:
            raise ValueError("incompatible query and candidate shape")
        if candidates.shape[-1] != DIMS:
            raise ValueError("candidate dimension mismatch")
        delta = candidates - query.unsqueeze(1)
        return -(self.projection(delta).square().sum(-1))


def state_hash(model: nn.Module) -> str:
    h = hashlib.sha256()
    for key,value in model.state_dict().items():
        h.update(key.encode())
        h.update(bytes(value.detach().cpu().contiguous().view(torch.uint8).flatten().tolist()))
    return h.hexdigest()


def training_batch(seed: int, step: int) -> tuple[Tensor, Tensor, Tensor]:
    # No evaluation namespace or evaluation seed appears in this generator.
    samples: list[list[tuple[float, ...]]] = []
    queries: list[tuple[float, ...]] = []
    targets: list[int] = []
    for j in range(TRAIN_BATCH):
        hard = (0.0, 0.25, 0.5, 1.0)[(step+j)%4]
        noise = (0.08, 0.25)[(step+j)%2]
        pair = make_pair(TRAIN_NAMESPACE + seed*1_000_003 + step*101 + j,
                         32, 16, hard, noise)
        ep = pair.left
        view = visible_inputs(ep)
        samples.append([x["address"] for x in (view["history"] + view["writes"])])
        queries.append(view["query_address"])
        # This supervision target is never passed into model.forward().
        targets.append(ep.target_slot)
    return (
        torch.tensor(queries, dtype=torch.float32),
        torch.tensor(samples, dtype=torch.float32),
        torch.tensor(targets, dtype=torch.long),
    )


def train_seed(seed: int, *, steps: int = TRAIN_STEPS) -> tuple[LearnedNoisyMetric, dict]:
    if steps < 1 or steps > TRAIN_STEPS:
        raise ValueError("invalid development training step count")
    torch.manual_seed(seed)
    model = LearnedNoisyMetric()
    initial = state_hash(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=0.0001)
    losses = []
    gradient_nonzero = False
    for step in range(steps):
        query, candidates, label = training_batch(seed, step)
        loss = F.cross_entropy(model(query,candidates), label)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        g = model.projection.weight.grad
        if g is not None and torch.isfinite(g).all() and float(g.abs().sum()) > 0:
            gradient_nonzero = True
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        losses.append(float(loss.detach()))
    final = state_hash(model)
    return model.eval(), {
        "training_seed": seed, "steps": steps,
        "initial_parameters_sha256": initial,
        "final_parameters_sha256": final,
        "parameters_changed": final != initial,
        "nonzero_finite_gradient": gradient_nonzero,
        "first10_mean_loss": sum(losses[:10])/len(losses[:10]),
        "last10_mean_loss": sum(losses[-10:])/len(losses[-10:]),
    }


def bucket(address: Sequence[float], capacity: int = CACHE_BUCKETS) -> int:
    """Content-derived fixed-capacity slot; no oracle identity or target slot."""
    if capacity != 64:
        raise ValueError("only registered development cache capacity 64 supported")
    b = 0
    for i,v in enumerate(address[:6]):
        if v >= 0:
            b |= 1 << i
    return b


@dataclass(frozen=True)
class CacheTrace:
    entries: tuple[dict, ...]
    build_writes: int
    intervening_writes: int
    overwrites: int
    target_retained: bool  # computed evaluator-side ONLY


def bounded_cache(ep: Episode, *, include_writes: bool) -> CacheTrace:
    """A genuine 64-slot overwriting cache; not learned/recurrent state."""
    view = visible_inputs(ep)
    slots: list[dict | None] = [None] * CACHE_BUCKETS
    overwrites = 0
    history = view["history"]
    writes = view["writes"] if include_writes else []
    for row in history+writes:
        b = bucket(row["address"])
        if slots[b] is not None:
            overwrites += 1
        slots[b] = row
    # Evaluator-only survival audit. target_slot is not consulted to
    # choose a slot or read a payload.
    item = history[ep.target_slot]
    retained = slots[bucket(item["address"])] is item
    return CacheTrace(
        entries=tuple(x for x in slots if x is not None),
        build_writes=len(history), intervening_writes=len(writes),
        overwrites=overwrites, target_retained=retained,
    )


@torch.no_grad()
def select_model_visible(model: LearnedNoisyMetric, ep: Episode,
                         *, cache: CacheTrace | None = None) -> tuple[int,int]:
    """Score visible query over visible entries; returns index and value."""
    v = visible_inputs(ep)
    items = (v["history"]+v["writes"]) if cache is None else list(cache.entries)
    q = torch.tensor([v["query_address"]], dtype=torch.float32)
    k = torch.tensor([[x["address"] for x in items]], dtype=torch.float32)
    idx = int(model(q,k).argmax(dim=1).item())
    return idx, items[idx]["value"] ^ v["current_bit"]


def dense_l2(ep: Episode) -> tuple[int,int]:
    v = visible_inputs(ep)
    entries = v["history"]+v["writes"]
    distances = [sum((a-b)**2 for a,b in zip(v["query_address"],x["address"]))
                 for x in entries]
    index = min(range(len(distances)),key=lambda i:distances[i])
    return index, entries[index]["value"] ^ v["current_bit"]


def evaluate_seed(model: LearnedNoisyMetric, seed: int,
                  *, pairs_per_cell: int=EVAL_PAIRS_PER_CELL,
                  H: Sequence[int]=(32,128,512),
                  D: Sequence[int]=(1,16,32),
                  fractions: Sequence[float]=(0.0,0.5,1.0),
                  noises: Sequence[float]=(0.08,0.25)) -> list[dict]:
    if pairs_per_cell < 1: raise ValueError("pairs must be positive")
    cells=[]
    for h in H:
        for d in D:
            for hard in fractions:
                for noise in noises:
                    c={"evaluation_seed":seed,"H":h,"D":d,
                       "hard_fraction":hard,"noise":noise,
                       "pair_count":pairs_per_cell,"episode_count":2*pairs_per_cell}
                    counts={k:0 for k in ("learned_correct","euclidean_correct",
                        "learned_recall_at_1","euclidean_recall_at_1",
                        "bounded_correct","bounded_nowrite_correct",
                        "bounded_retained","bounded_nowrite_retained",
                        "bounded_swapped_correct","reset_correct")}
                    overwrite_sum=0
                    cache_probes=0
                    all_candidate_scores=0
                    for j in range(pairs_per_cell):
                        dataset_seed=(EVAL_NAMESPACE + seed*1_000_003
                                      + h*10_007 + d*3_007
                                      + int(hard*100)*401
                                      + int(noise*100)*37 + j)
                        pair=make_pair(dataset_seed,h,d,hard,noise)
                        ep_list=(pair.left,pair.right)
                        traces=[bounded_cache(ep,include_writes=True) for ep in ep_list]
                        for ei,ep in enumerate(ep_list):
                            v=visible_inputs(ep)
                            target=ep.target
                            learned_i,learned_y=select_model_visible(model,ep)
                            raw_i,raw_y=dense_l2(ep)
                            yes=traces[ei]
                            no=bounded_cache(ep,include_writes=False)
                            _,cache_y=select_model_visible(model,ep,cache=yes)
                            _,nowrite_y=select_model_visible(model,ep,cache=no)
                            # Swap memory content between opposite-past partners
                            # but retain the same present/query observation.
                            _,swapped_y=select_model_visible(
                                model,ep,cache=traces[1-ei])
                            counts["learned_correct"]+= learned_y==target
                            counts["euclidean_correct"]+= raw_y==target
                            counts["learned_recall_at_1"]+= learned_i==ep.target_slot
                            counts["euclidean_recall_at_1"]+= raw_i==ep.target_slot
                            counts["bounded_correct"]+= cache_y==target
                            counts["bounded_nowrite_correct"]+= nowrite_y==target
                            counts["bounded_retained"]+= yes.target_retained
                            counts["bounded_nowrite_retained"]+= no.target_retained
                            counts["bounded_swapped_correct"]+= swapped_y==target
                            counts["reset_correct"]+= v["current_bit"]==target
                            overwrite_sum+=yes.overwrites
                            cache_probes+=len(yes.entries)
                            all_candidate_scores+=len(v["history"])+len(v["writes"])
                    n=c["episode_count"]
                    c.update({k:v/n for k,v in counts.items()})
                    c["mean_cache_overwrites"]=overwrite_sum/n
                    c["mean_bounded_query_candidates"]=cache_probes/n
                    c["dense_query_candidates"]=all_candidate_scores/n
                    c["cache_build_plus_refresh"]=h+d
                    c["cache_capacity"]=CACHE_BUCKETS
                    cells.append(c)
    return cells
