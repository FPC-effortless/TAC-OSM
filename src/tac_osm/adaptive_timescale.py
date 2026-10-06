"""End-to-end trainable temporal decay mechanism for TDBU research."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np

from tac_osm.mtsk_topdown import FixedBinaryExecutor, IndependentVerifier, environment_outcome

EXPERIMENT_ID = "TACOSM-PLM-TDBU-ADAPTIVE-TIMESCALE-001"
GENERATOR_VERSION = "adaptive-timescale-v1-train0.30-0.985-test0.41-0.994"
BENCHMARK_HASH = hashlib.sha256(GENERATOR_VERSION.encode()).hexdigest()

TRAIN_FILTER_ALPHAS = np.array((0.30,0.50,0.68,0.82,0.90,0.96,0.985),dtype=np.float64)
TEST_FILTER_ALPHAS = np.array((0.41,0.57,0.76,0.875,0.925,0.991,0.994),dtype=np.float64)
PAIR_TYPES_TRAIN = tuple((i,j) for i in range(len(TRAIN_FILTER_ALPHAS)) for j in range(i+1,len(TRAIN_FILTER_ALPHAS)))
PAIR_TYPES_TEST = tuple((i,j) for i in range(len(TEST_FILTER_ALPHAS)) for j in range(i+1,len(TEST_FILTER_ALPHAS))
)
PARENT_ALPHAS = np.array((0.50,0.90,0.98),dtype=np.float64)
ARMS = ("no_state","one_fixed","three_fixed","three_adaptive")
POLICY_HIDDEN = {"no_state":12,"one_fixed":12,"three_fixed":12,"three_adaptive":12}

Arm = Literal["no_state","one_fixed","three_fixed","three_adaptive"]

@dataclass(frozen=True)
class EpisodeExample:
    history: np.ndarray
    label: int
    pair_id: int
    current_observation: float = 0.0

def _filter(history: np.ndarray, alpha: float) -> float:
    state=0.0
    for value in history:
        state=alpha*state+(1.0-alpha)*float(value)
    return state

def target_action(history: np.ndarray, pair_type: tuple[int,int], alphas: np.ndarray) -> int:
    left=_filter(history,float(alphas[pair_type[0]]))
    right=_filter(history,float(alphas[pair_type[1]]))
    return int(left>=right)

def make_pairs(seed:int,history_length:int,n_pairs:int,*,alphas:np.ndarray,pair_types:Sequence[tuple[int,int]]) -> list[EpisodeExample]:
    if history_length<1 or n_pairs<1: raise ValueError("history_length and n_pairs must be positive")
    rng=np.random.default_rng(seed)
    out=[]
    for pair_id in range(n_pairs):
        pair_type=pair_types[pair_id%len(pair_types)]
        history=rng.normal(0.0,1.0,size=history_length).astype(np.float64)
        if target_action(history,pair_type,alphas)!=1: history=-history
        out.append(EpisodeExample(history.copy(),1,pair_id))
        out.append(EpisodeExample(-history,0,pair_id))
    return out

def _sigmoid(x:np.ndarray)->np.ndarray:
    return 1.0/(1.0+np.exp(-np.clip(x,-30.0,30.0)))

def _logit(p:np.ndarray)->np.ndarray:
    p=np.clip(np.asarray(p,dtype=np.float64),1e-6,1.0-1e-6)
    return np.log(p/(1.0-p))

class TemporalState:
    def __init__(self, arm:Arm, alpha_values:np.ndarray|None=None):
        self.arm=arm
        if arm=="no_state": self.alphas=np.zeros(3,dtype=np.float64)
        elif arm=="one_fixed": self.alphas=np.array((0.90,0.0,0.0),dtype=np.float64)
        elif arm=="three_fixed": self.alphas=PARENT_ALPHAS.copy()
        elif arm=="three_adaptive":
            self.alphas=np.array(alpha_values if alpha_values is not None else PARENT_ALPHAS,dtype=np.float64)
        else: raise ValueError(arm)
        self.reset()

    def reset(self)->None:
        self.values=np.zeros(3,dtype=np.float64)

    def process(self,history:np.ndarray)->np.ndarray:
        self.reset()
        for x in history:
            self.values[0]=self.alphas[0]*self.values[0]+(1.0-self.alphas[0])*float(x)
            self.values[1]=self.alphas[1]*self.values[1]+(1.0-self.alphas[1])*float(x)
            self.values[2]=self.alphas[2]*self.values[2]+(1.0-self.alphas[2])*float(x)
        return self.values.copy()

    def process_with_derivatives(self,history:np.ndarray)->tuple[np.ndarray,np.ndarray]:
        if self.arm!="three_adaptive": return self.process(history),np.zeros(3,dtype=np.float64)
        self.reset()
        deriv=np.zeros(3,dtype=np.float64)
        for x in history:
            prev=self.values.copy()
            self.values=self.alphas*self.values+(1.0-self.alphas)*float(x)
            deriv=prev+self.alphas*deriv-float(x)
        return self.values.copy(),deriv

class AdaptiveTemporalPolicy:
    def __init__(self,seed:int):
        rng=np.random.default_rng(seed)
        self.W1=rng.normal(0.0,0.5,size=(4,12))
        self.b1=np.zeros(12,dtype=np.float64)
        self.W2=rng.normal(0.0,0.5,size=(12,2))
        self.b2=np.zeros(2,dtype=np.float64)
        self.alpha_logits=_logit(PARENT_ALPHAS)

    @property
    def parameter_count(self)->int:
        return int(self.W1.size+self.b1.size+self.W2.size+self.b2.size)

    @property
    def alpha_count(self)->int:
        return 3

    @property
    def alphas(self)->np.ndarray:
        return _sigmoid(self.alpha_logits)

    def _forward(self,X:np.ndarray)->tuple[np.ndarray,np.ndarray,np.ndarray]:
        z=X@self.W1+self.b1
        h=np.tanh(z)
        logits=h@self.W2+self.b2
        shifted=logits-logits.max(axis=1,keepdims=True)
        e=np.exp(shifted)
        probs=e/e.sum(axis=1,keepdims=True)
        return h,logits,probs

    def fit(self,examples:list[EpisodeExample],steps:int,policy_lr:float,alpha_lr:float)->int:
        y=np.asarray([e.label for e in examples],dtype=np.int64)
        for _ in range(steps):
            X=np.zeros((len(examples),4),dtype=np.float64)
            dstate=np.zeros((len(examples),3),dtype=np.float64)
            state=TemporalState("three_adaptive",self.alphas)
            for i,e in enumerate(examples):
                X[i,0]=e.current_observation
                X[i,1:],dstate[i]=state.process_with_derivatives(e.history)
            h,_logits,probs=self._forward(X)
            yoh=np.eye(2,dtype=np.float64)[y]
            dlogits=(probs-yoh)/len(X)
            dW2=h.T@dlogits
            db2=dlogits.sum(axis=0)
            dh=dlogits@self.W2.T
            dz=dh*(1.0-h*h)
            dW1=X.T@dz
            db1=dz.sum(axis=0)
            dX=dz@self.W1.T
            dalpha=np.sum(dX[:,1:]*dstate,axis=0)
            self.W2-=policy_lr*dW2
            self.b2-=policy_lr*db2
            self.W1-=policy_lr*dW1
            self.b1-=policy_lr*db1
            sigmoid=self.alphas
            self.alpha_logits-=alpha_lr*dalpha*sigmoid*(1.0-sigmoid)
        return steps

class FixedTemporalPolicy(AdaptiveTemporalPolicy):
    def __init__(self,seed:int):
        super().__init__(seed)
    def fit(self,examples:list[EpisodeExample],steps:int,policy_lr:float,alpha_lr:float)->int:
        X=np.zeros((len(examples),4),dtype=np.float64)
        state=TemporalState("three_fixed",PARENT_ALPHAS)
        for i,e in enumerate(examples): X[i,0]=e.current_observation; X[i,1:]=state.process(e.history)
        y=np.asarray([e.label for e in examples],dtype=np.int64)
        for _ in range(steps):
            h,_logits,probs=self._forward(X)
            yoh=np.eye(2,dtype=np.float64)[y]
            dlogits=(probs-yoh)/len(X)
            dW2=h.T@dlogits; db2=dlogits.sum(axis=0)
            dh=dlogits@self.W2.T; dz=dh*(1.0-h*h)
            dW1=X.T@dz; db1=dz.sum(axis=0)
            self.W2-=policy_lr*dW2; self.b2-=policy_lr*db2
            self.W1-=policy_lr*dW1; self.b1-=policy_lr*db1
        return steps

def make_policy(seed:int,arm:Arm)->AdaptiveTemporalPolicy:
    return AdaptiveTemporalPolicy(seed=seed)

def _features(examples:list[EpisodeExample],arm:Arm,intervention:str="normal",alpha_values:np.ndarray|None=None)->np.ndarray:
    state=TemporalState(arm,alpha_values)
    rows=[]
    for e in examples: rows.append(np.concatenate(([e.current_observation],state.process(e.history))))
    X=np.asarray(rows,dtype=np.float64)
    if intervention=="reset": X[:,1:]=0.0
    elif intervention=="shuffle": X[:,1:]=X[np.arange(len(X))[::-1],1:]
    elif intervention!="normal": raise ValueError(intervention)
    return X

def evaluate(policy:AdaptiveTemporalPolicy,examples:list[EpisodeExample],arm:Arm,*,intervention:str="normal",alpha_values:np.ndarray|None=None)->dict[str,float]:
    if arm=="three_adaptive": alpha_values=policy.alphas.copy()
    X=_features(examples,arm,intervention,alpha_values)
    pred=policy.predict(X)
    labels=np.asarray([e.label for e in examples],dtype=np.int64)
    executor=FixedBinaryExecutor(); verifier=IndependentVerifier()
    executed=np.asarray([executor.execute(int(a)) for a in pred],dtype=np.int64)
    outcomes=np.asarray([environment_outcome(int(a),int(y)) for a,y in zip(executed,labels)],dtype=np.float64)
    verified=np.asarray([verifier.verify(int(a),int(y)) for a,y in zip(executed,labels)],dtype=np.float64)
    if not np.array_equal(outcomes,verified): raise RuntimeError("outcome/verifier mismatch")
    pair_predictions={}
    for e,p in zip(examples,pred): pair_predictions.setdefault(e.pair_id,[]).append(int(p))
    return {"success":float(np.mean(verified)),"action_gap":float(np.mean([len(v)==2 and v[0]!=v[1] for v in pair_predictions.values()]))}

def _predict(self,X): return np.argmax(self._forward(X)[2],axis=1)
AdaptiveTemporalPolicy.predict=_predict

def parameter_count()->int: return 86
def benchmark_hash()->str: return BENCHMARK_HASH
