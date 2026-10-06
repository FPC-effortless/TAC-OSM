"""Single-relation temporal-memory benchmark and state arms."""
from __future__ import annotations
import hashlib
from dataclasses import dataclass
import numpy as np
from tac_osm.mtsk_topdown import FixedBinaryExecutor, IndependentVerifier, MLPPolicy, environment_outcome

EXPERIMENT_ID="TACOSM-PLM-TDBU-MEMORY-ISOLATION-001"
GENERATOR_VERSION="memory-isolation-v1-ground-truth-0.45-0.985"
BENCHMARK_HASH=hashlib.sha256(GENERATOR_VERSION.encode()).hexdigest()
GROUND_TRUTH_ALPHAS=(0.45,0.985)
STATE_ALPHAS={
 "no_state":(None,None,None),
 "single_timescale":(0.90,None,None),
 "two_timescale":(0.50,0.98,None),
 "mtsk":(0.50,0.90,0.98),
}
ARMS=("no_state","single_timescale","two_timescale","mtsk")

@dataclass(frozen=True)
class EpisodeExample:
    history:np.ndarray
    label:int
    pair_id:int
    current_observation:float=0.0

def _filter(history:np.ndarray,alpha:float)->float:
    s=0.0
    for x in history:
        s=alpha*s+(1.0-alpha)*float(x)
    return s

def target_action(history:np.ndarray)->int:
    return int(_filter(history,GROUND_TRUTH_ALPHAS[0])>=_filter(history,GROUND_TRUTH_ALPHAS[1]))

def make_pairs(seed:int,history_length:int,n_pairs:int)->list[EpisodeExample]:
    if history_length<1 or n_pairs<1: raise ValueError("history_length and n_pairs must be positive")
    rng=np.random.default_rng(seed); out=[]
    for pair_id in range(n_pairs):
        h=rng.normal(0.0,1.0,size=history_length).astype(np.float64)
        if target_action(h)!=1: h=-h
        out.append(EpisodeExample(h.copy(),1,pair_id))
        out.append(EpisodeExample(-h,0,pair_id))
    return out

def featurize(examples:list[EpisodeExample],arm:str)->np.ndarray:
    alphas=STATE_ALPHAS[arm]
    rows=[]
    for e in examples:
        vals=[]
        for alpha in alphas:
            vals.append(0.0 if alpha is None else _filter(e.history,float(alpha)))
        rows.append(np.concatenate(([e.current_observation],np.asarray(vals,dtype=np.float64))))
    return np.asarray(rows,dtype=np.float64)

def representability_sign_accuracy(seed:int=12345,n_pairs:int=3000)->float:
    ex=make_pairs(seed,64,n_pairs)
    X=featurize(ex,"mtsk")
    y=np.asarray([e.label for e in ex],dtype=np.int64)
    pred=(X[:,1]>=X[:,3]).astype(int)
    return float(np.mean(pred==y))

def evaluate(policy:MLPPolicy,examples:list[EpisodeExample],arm:str,intervention:str="normal")->dict[str,float]:
    X=featurize(examples,arm)
    if intervention=="reset":
        X[:,1:]=0.0
    elif intervention=="shuffle":
        X[:,1:]=X[np.arange(len(X))[::-1],1:]
    elif intervention!="normal":
        raise ValueError(intervention)
    pred=policy.predict(X)
    y=np.asarray([e.label for e in examples],dtype=np.int64)
    executor=FixedBinaryExecutor()
    verifier=IndependentVerifier()
    actions=np.asarray([executor.execute(int(a)) for a in pred],dtype=np.int64)
    outcomes=np.asarray([environment_outcome(int(a),int(t)) for a,t in zip(actions,y)],dtype=np.float64)
    verified=np.asarray([verifier.verify(int(a),int(t)) for a,t in zip(actions,y)],dtype=np.float64)
    if not np.array_equal(outcomes,verified):
        raise RuntimeError("verifier mismatch")
    by={}
    for e,p in zip(examples,pred):
        by.setdefault(e.pair_id,[]).append(int(p))
    gap=float(np.mean([len(v)==2 and v[0]!=v[1] for v in by.values()]))
    state_ops={"no_state":0,"single_timescale":3,"two_timescale":6,"mtsk":9}[arm]
    work=float(len(examples[0].history)*state_ops + 6*policy.hidden + 1)
    return {"success":float(np.mean(verified)),"action_gap":gap,"work":work}

def benchmark_hash()->str:
    return BENCHMARK_HASH

def dependency_hash()->str:
    return hashlib.sha256(b"numpy-only-memory-isolation-v1").hexdigest()
