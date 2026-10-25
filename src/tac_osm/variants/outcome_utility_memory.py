"""EWA001: causal bounded external residual memory with cheap LSH addressing.

Experimental adjunct to FROZEN original ME097; NEVER part of its 8x8 scan.
All outcomes arrive only in observe(), after a prediction token is produced.
All 256 slots (8 key + 14 normalized real residual + 1 utility) are
additional floats; index/IDs/ages/Python allocations are additional bytes.
"""
from __future__ import annotations
from dataclasses import dataclass
import math
import time
from typing import Optional
import numpy as np

DIM=8
OUTPUT=14
CAPACITY=256
TOP_K=8
PLANES=8
SEED=20261025
CAP=0.08
CORRECTION=0.25
UTILITY_WEIGHT=0.20
EMA=0.05

@dataclass(frozen=True)
class Pending:
    query_id: int
    base: np.ndarray
    forecast: np.ndarray
    selected: tuple[tuple[int,int], ...]
    candidates_scored: int
    bucket_probes: int
    hit: bool
    recall: float | None

@dataclass
class Item:
    key: np.ndarray
    residual: np.ndarray
    utility: float
    serial: int
    generation: int
    source: str
    signature: int

class OnlineMemory:
    def __init__(self, mode: str, capacity: int=CAPACITY):
        if mode not in ("neutral","utility"):
            raise ValueError("only neutral and utility memory modes")
        if capacity<1:
            raise ValueError("capacity must be positive")
        self.mode=mode
        self.capacity=capacity
        self.slots: list[Optional[Item]]=[None]*capacity
        self.buckets: dict[int,set[int]]={}
        self.planes=np.random.default_rng(SEED).normal(size=(PLANES,DIM)).astype(np.float32)
        self.serial=0
        self.generation=0
        self.next_query=0
        self.pending: Optional[Pending]=None
        self.queries=0
        self.buckets_probed=0
        self.candidates_scored=0
        self.recall_sum=0.0
        self.recall_trials=0
        self.hits=0
        self.replacements=0
        self.utility_updates=0
        self.build_cpu=0.0
        self.lookup_cpu=0.0

    @staticmethod
    def normalized(k):
        x=np.asarray(k,dtype=np.float32).reshape(-1)
        if x.shape!=(DIM,) or not np.isfinite(x).all():
            raise ValueError("real model query must be finite eight dimensional")
        n=float(np.linalg.norm(x))
        return x / max(n,1e-9)

    @staticmethod
    def vec(x):
        y=np.asarray(x,dtype=np.float32).reshape(-1)
        if y.shape!=(OUTPUT,) or not np.isfinite(y).all():
            raise ValueError("finite fourteen-vector required")
        return y.copy()

    def _sig(self, key):
        dots=self.planes@key
        signature=0
        for i,p in enumerate(dots):
            if p>=0:signature|=1<<i
        return signature

    def _candidate_indices(self, sig):
        positions={sig}
        for j in range(PLANES):
            positions.add(sig^(1<<j))
        found=set()
        for p in positions:
            found.update(self.buckets.get(p,()))
        return sorted(found),len(positions)

    def predict(self, key, baseline, *, audit: bool=False):
        if self.pending is not None:
            raise RuntimeError("outcome of previous real event must be verified first")
        q=self.normalized(key)
        base=self.vec(baseline)
        ts=time.process_time()
        choices,nprobes=self._candidate_indices(self._sig(q))
        ranked=[]
        for i in choices:
            m=self.slots[i]
            if m is None:
                raise AssertionError("stale LSH location")
            score=float(np.dot(q,m.key))
            if self.mode=="utility":
                score+=UTILITY_WEIGHT*m.utility
            ranked.append((-score,i))
        ranked.sort()
        indices=[i for _,i in ranked[:TOP_K]]
        prediction=base.copy()
        if indices:
            # Non-parametric correction, same strength and cap across arms.
            scores=np.asarray([-score for score,i in ranked[:TOP_K]],dtype=np.float64)
            probs=np.exp((scores-scores.max())/.25)
            probs/=probs.sum()
            correction=sum(float(w)*self.slots[i].residual for i,w in zip(indices,probs))
            prediction=(base+CORRECTION*np.clip(correction,-CAP,CAP)).astype(np.float32)
        recall=None
        if audit and any(item is not None for item in self.slots):
            # Evaluation-only top8 brute reference, never supplied to controller.
            full=[(-float(np.dot(q,m.key)),i) for i,m in enumerate(self.slots) if m is not None]
            full.sort()
            nearest=set(i for _,i in full[:TOP_K])
            recall=len(set(indices)&nearest)/max(1,len(nearest))
            self.recall_sum+=recall
            self.recall_trials+=1
        self.next_query+=1
        self.queries+=1
        self.buckets_probed+=nprobes
        self.candidates_scored+=len(choices)
        self.hits+=int(bool(indices))
        self.lookup_cpu+=time.process_time()-ts
        token=Pending(self.next_query,base,prediction,tuple((i,self.slots[i].generation) for i in indices),
                      len(choices),nprobes,bool(indices),recall)
        self.pending=token
        return token

    def observe(self, token:Pending, real_next, key, source):
        if self.pending is not token:
            raise RuntimeError("verified next observation must follow its issued prediction")
        y=self.vec(real_next)
        k=self.normalized(key)
        if not isinstance(source,str) or not source:
            raise ValueError("authenticated event source must be supplied")
        ts=time.process_time()
        # A verified event may update utility only AFTER its previous forecast.
        if self.mode=="utility":
            original_loss=float(np.mean((token.base-y)**2))
            for slot,generation in token.selected:
                item=self.slots[slot]
                if item is None or item.generation!=generation:
                    raise RuntimeError("memory replaced before delayed verification")
                individual=(token.base+CORRECTION*np.clip(item.residual,-CAP,CAP)).astype(np.float32)
                corrected_loss=float(np.mean((individual-y)**2))
                advantage=np.clip((original_loss-corrected_loss)/max(original_loss,1e-6),-1,1)
                item.utility=float(np.clip((1-EMA)*item.utility+EMA*advantage,-1,1))
                self.utility_updates+=1
        residual=np.clip(y-token.base,-CAP,CAP).astype(np.float32)
        vacancy=next((i for i,m in enumerate(self.slots) if m is None),None)
        if vacancy is not None:
            i=vacancy
        else:
            self.replacements+=1
            if self.mode=="neutral":
                i=min(range(self.capacity),key=lambda j:(self.slots[j].serial,j))
            else:
                # State-dependent bounded online eviction. Age prevents stale immortal records.
                newest=max(x.serial for x in self.slots)
                i=min(range(self.capacity),key=lambda j:(
                    self.slots[j].utility - .10*(newest-self.slots[j].serial)/self.capacity,j))
            old=self.slots[i]
            self.buckets[old.signature].remove(i)
            if not self.buckets[old.signature]:del self.buckets[old.signature]
        sig=self._sig(k)
        self.serial+=1
        self.generation+=1
        self.slots[i]=Item(k.copy(),residual.copy(),0.0,self.serial,self.generation,source,sig)
        self.buckets.setdefault(sig,set()).add(i)
        self.pending=None
        self.build_cpu+=time.process_time()-ts

    def stats(self):
        if self.pending is not None:raise RuntimeError("pending unverified prediction")
        occupied=sum(x is not None for x in self.slots)
        return dict(mode=self.mode,capacity=self.capacity,occupied=occupied,
            float_payload_bytes=self.capacity*(DIM+OUTPUT+1)*4,
            extra_index_and_metadata_bytes="NOT_IN_608B_UNMEASURED_PYTHON_ALLOCATIONS",
            queries=self.queries,matching_candidate_scores=self.candidates_scored,
            candidate_scores_per_query=self.candidates_scored/max(1,self.queries),
            bucket_probes_per_query=self.buckets_probed/max(1,self.queries),
            fraction_with_candidates=self.hits/max(1,self.queries),
            top8_exact_reference_recall=self.recall_sum/self.recall_trials if self.recall_trials else None,
            measured_lookup_cpu_seconds=self.lookup_cpu,measured_write_cpu_seconds=self.build_cpu,
            replacements=self.replacements,utility_updates=self.utility_updates)
