"""EWA002: separate fixed-slot real-event memory, hash vs brute, delayed utility.

NOT a trained neural model; preserves original ME097 input-only scan untouched.
Two cosine routes use identical FIFO bank; utility reranks within SAME hashed
candidates. Utility tracks *leave-one-retrieval-out* change in actual verified
forecast loss, AFTER the previously issued outcome has arrived.
"""
from __future__ import annotations
from dataclasses import dataclass
import time
import numpy as np

DIM=8
VALUE=14
CAPACITY=256
K=8
BITS=11
RADIUS=1
LSH_SEED=20261025
CAP=0.08
GAIN=0.25
TEMPERATURE=0.25
UTILITY_WEIGHT=0.20
UTILITY_EMA=0.05

@dataclass(frozen=True)
class MemoryQuery:
    velocity: np.ndarray
    exhaustive: np.ndarray
    hashed: np.ndarray
    utility: np.ndarray
    selected: tuple[int,...]
    selected_weights: tuple[float,...]
    size: int
    collisions: int
    hash_recall: float
    hash_hit: bool
    exhaustive_top8: tuple[int,...]
    hashed_top8: tuple[int,...]

class VelocityResidualMemory:
    def __init__(self, scale:float):
        if not np.isfinite(scale) or scale<=0:
            raise ValueError("real FIT-only residual loss scale must be positive finite")
        self.scale=float(scale)
        self.keys=np.zeros((CAPACITY,DIM),dtype=np.float32)
        self.residual=np.zeros((CAPACITY,VALUE),dtype=np.float32)
        self.priority=np.zeros((CAPACITY,),dtype=np.float32)
        self.signature=np.zeros((CAPACITY,),dtype=np.int32)
        self.provenance=[""]*CAPACITY
        self.buckets:dict[int,set[int]]={}
        self.n=0
        self.cursor=0
        self.planes=np.random.default_rng(LSH_SEED).normal(size=(BITS,DIM)).astype(np.float32)
        self.pending:MemoryQuery|None=None
        self.query_count=0
        self.hashed_scored=0
        self.exhaustive_scored=0
        self.hash_recall_total=0.0
        self.hash_hit_total=0
        self.changed_utility=0
        self.hash_probe_count=0
        self.read_cpu=0.0
        self.write_cpu=0.0

    @staticmethod
    def normalize(q):
        q=np.asarray(q,dtype=np.float32)
        if q.shape!=(DIM,) or not np.isfinite(q).all():
            raise ValueError("missing or nonfinite 8D original learned query")
        norm=float(np.linalg.norm(q))
        return q/max(norm,1e-8)

    @staticmethod
    def vec(v):
        v=np.asarray(v,dtype=np.float32)
        if v.shape!=(VALUE,) or not np.isfinite(v).all():
            raise ValueError("physical 14D vector missing/nonfinite")
        return v.copy()

    def _signature(self,key):
        dots=self.planes@key
        v=0
        for i,x in enumerate(dots):
            if x>=0:v|=1<<i
        return v

    @staticmethod
    def _top(scores,indices):
        if len(indices)==0:return np.empty(0,dtype=np.int64)
        # Stable, deterministic ties, score desc / index asc.
        ranked=sorted(((-float(score),int(i)) for score,i in zip(scores,indices)))
        return np.asarray([i for _,i in ranked[:K]],dtype=np.int64)

    def _forecast(self,base,ids,scores):
        if len(ids)==0:return base.copy(),np.zeros(0,dtype=np.float64)
        # Same normalized weighting independent of choice of cache route.
        logits=np.asarray(scores,dtype=np.float64)/TEMPERATURE
        z=np.exp(logits-logits.max());z/=z.sum()
        correction=np.sum(z[:,None]*self.residual[ids],axis=0)
        out=base+GAIN*np.clip(correction,-CAP,CAP)
        return out.astype(np.float32),z

    def predict(self,query,base):
        if self.pending is not None:
            raise RuntimeError("cannot issue next memory query before real outcome verification")
        q=self.normalize(query)
        x=self.vec(base)
        ts=time.process_time()
        full_ids=np.arange(self.n,dtype=np.int64)
        full_scores=(self.keys[:self.n]@q).astype(np.float32)
        full=self._top(full_scores,full_ids)
        fullpred,_=self._forecast(x,full,full_scores[full])
        sig=self._signature(q)
        probes=(sig,)+tuple(sig^(1<<j) for j in range(BITS))
        found:set[int]=set()
        for p in probes:
            found.update(self.buckets.get(p,set()))
        ids=np.asarray(sorted(found),dtype=np.int64)
        h_scores=full_scores[ids]
        htop=self._top(h_scores,ids)
        hpred,_=self._forecast(x,htop,full_scores[htop])
        util_scores=full_scores[ids]+UTILITY_WEIGHT*self.priority[ids]
        selected=self._top(util_scores,ids)
        useful_pred,weight=self._forecast(x,selected,full_scores[selected]+UTILITY_WEIGHT*self.priority[selected])
        recall=len(set(full.tolist())&set(ids.tolist()))/len(full) if len(full) else 1.0
        record=MemoryQuery(x,fullpred,hpred,useful_pred,tuple(int(i) for i in selected),
                           tuple(float(w) for w in weight),len(ids),self.n,
                           float(recall),bool(len(ids)),tuple(int(i) for i in full),tuple(int(i) for i in htop))
        self.pending=record
        self.query_count+=1
        self.hashed_scored+=len(ids)
        self.exhaustive_scored+=self.n
        self.hash_recall_total+=recall
        self.hash_hit_total+=int(bool(len(ids)))
        self.hash_probe_count+=len(probes)
        self.read_cpu+=time.process_time()-ts
        return record

    def observe(self,record:MemoryQuery,real_next,query,provenance):
        if self.pending is not record:
            raise RuntimeError("observation must verify latest issued prediction")
        y=self.vec(real_next)
        key=self.normalize(query)
        if not isinstance(provenance,str) or not provenance:
            raise ValueError("source/frame provenance required")
        ts=time.process_time()
        # Only actual prior-retrieved memory entries receive later observation credit.
        # Leave-one-out: remove current record i and renormalize other weights,
        # then compute its observed effect on the same ensemble prediction.
        if len(record.selected):
            selected=np.asarray(record.selected,dtype=np.int64)
            weights=np.asarray(record.selected_weights,dtype=np.float64)
            correction=self.residual[selected].astype(np.float64)
            with_loss=float(np.mean((record.utility-y)**2))
            for j,i in enumerate(selected):
                if len(selected)==1:
                    wo=record.velocity
                else:
                    other_weights=np.delete(weights,j)
                    other_weights/=other_weights.sum()
                    other=np.delete(correction,j,axis=0)
                    delta=np.sum(other_weights[:,None]*other,axis=0)
                    wo=(record.velocity+GAIN*np.clip(delta,-CAP,CAP)).astype(np.float32)
                without_loss=float(np.mean((wo-y)**2))
                change=float(np.clip((without_loss-with_loss)/self.scale,-1,1))
                self.priority[i]=float(np.clip((1-UTILITY_EMA)*self.priority[i]+UTILITY_EMA*change,-1,1))
                self.changed_utility+=1
        # Add this now-observed genuine residual into common FIFO bank.
        target=np.clip(y-record.velocity,-CAP,CAP).astype(np.float32)
        i=self.cursor
        if self.n==CAPACITY:
            prev_sig=int(self.signature[i])
            self.buckets[prev_sig].remove(i)
            if not self.buckets[prev_sig]:
                del self.buckets[prev_sig]
        else:
            self.n+=1
        s=self._signature(key)
        self.keys[i]=key
        self.residual[i]=target
        self.priority[i]=0.0
        self.signature[i]=s
        self.provenance[i]=provenance
        self.buckets.setdefault(s,set()).add(i)
        self.cursor=(self.cursor+1)%CAPACITY
        self.pending=None
        self.write_cpu+=time.process_time()-ts

    def counters(self):
        if self.pending is not None:
            raise RuntimeError("no audit statistics before pending real outcome")
        return dict(executed_queries=self.query_count,hash_candidate_scores=self.hashed_scored,
           exhaustive_candidate_scores=self.exhaustive_scored,
           indexed_candidates_per_query=self.hashed_scored/max(1,self.query_count),
           full_candidates_per_query=self.exhaustive_scored/max(1,self.query_count),
           hash_raw_top8_recall=self.hash_recall_total/max(1,self.query_count),
           hashed_nonempty_rate=self.hash_hit_total/max(1,self.query_count),
           hash_probes_per_query=self.hash_probe_count/max(1,self.query_count),
           utility_feedback_updates=self.changed_utility,
           separate_float_memory_bytes=CAPACITY*(DIM+VALUE+1)*4,
           total_runtime_bytes="GREATER; includes buckets/source IDs/runtime, NOT 608B",
           cpu_read_seconds=self.read_cpu,cpu_write_seconds=self.write_cpu,
           occupancy=self.n)
