"""Training/evaluation-only joint public posterior adapter; never exported."""
import numpy as np
import torch
from data_belief_dev062 import BeliefWorld,snapshot as health_snapshot,batch as health_batch


def relation(w):return 1 if w.parents[1]==0 else 2 if w.parents[0]==1 else 0


def snapshot(w,b):
    r=health_snapshot(w,b);graph=relation(w);r['health_posterior']=r['posterior'];joint=torch.zeros(4,3);joint[:,graph]=r['posterior'];r['posterior']=joint.flatten();r['truth']=r['truth']*3+graph;return r


def corpus(seed,count):
    records=[]
    for index in range(count):
        w=BeliefWorld(seed,index,(32,128)[index%2],inspection=(.05,.1,.2)[index%3]);b=w.reference();rng=np.random.default_rng(np.random.SeedSequence([seed,index,634]))
        for _ in range(int(rng.integers(7))):
            action=int(rng.integers(1,5)) if rng.random()<.4 else int(rng.choice(5,p=b.policy()))
            if action==0:break
            b.update(action,w.act(action))
        records.append(snapshot(w,b))
    return records


def batch(records,ids,gate):return health_batch(records,ids,gate)
