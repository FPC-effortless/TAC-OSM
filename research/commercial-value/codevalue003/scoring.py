"""Causal full-file scoring and free-running suggestions; no training code."""
from collections import Counter, defaultdict
from functools import lru_cache
import math
import time

import numpy as np
import torch

LENGTHS=(2,4,8,16,32)

def blocks(raw):
    # Each gold byte64..len-1 is scored exactly once. Prefix length64..127.
    for base in range(64,len(raw),64):
        stop=min(base+64,len(raw))
        yield list(raw[base-64:stop-1]),list(raw[base:stop])

def offers(raw):
    positions=np.linspace(128,len(raw)-32,512,dtype=np.int64).tolist()
    if len(set(positions))!=512:raise ValueError('source cannot supply512 unique offers')
    return positions

def initial():
    return dict(targets=0,nll_sum=0.,top1=0,top5=0,brier_sum=0.,
                calibration_counts=[0]*10,calibration_correct=[0]*10,
                calibration_confidence=[0.]*10)

def add_predictions(result,prob,y):
    # CPU NumPy scoring shared by neural and classical methods.
    y=np.asarray(y,dtype=np.int64)
    if prob.shape!=(len(y),256) or not np.isfinite(prob).all():raise ValueError('invalid probabilities')
    if not np.allclose(prob.sum(-1),1,atol=1e-5):raise ValueError('probabilities not normalized')
    best=prob.argmax(-1);conf=prob.max(-1);hit=(best==y)
    result['targets']+=len(y)
    result['nll_sum']+=float(-np.log(np.maximum(prob[np.arange(len(y)),y],1e-300)).sum())
    result['top1']+=int(hit.sum())
    result['top5']+=int((np.argpartition(prob,-5,axis=-1)[:,-5:]==y[:,None]).any(-1).sum())
    result['brier_sum']+=float((prob*prob).sum()-2*prob[np.arange(len(y)),y].sum()+len(y))
    bins=np.minimum((conf*10).astype(int),9)
    for i in range(10):
        idx=(bins==i)
        result['calibration_counts'][i]+=int(idx.sum())
        result['calibration_correct'][i]+=int(hit[idx].sum())
        result['calibration_confidence'][i]+=float(conf[idx].sum())

def finish(row):
    n=row['targets']
    row.update(nll=row['nll_sum']/n,bits_per_byte=row['nll_sum']/n/math.log(2),
               accuracy=row['top1']/n,top5_accuracy=row['top5']/n,brier=row['brier_sum']/n)
    row['ece10']=sum(abs(c-s) for c,s in zip(row['calibration_correct'],row['calibration_confidence']))/n
    return row

def suggestion_row(position,generated,target):
    prefix=0
    for a,b in zip(generated,target):
        if a!=b:break
        prefix+=1
    return {'position':position,'generated_hex':bytes(generated).hex(),'longest_correct_prefix':prefix,
            'exact':{str(n):int(prefix>=n) for n in LENGTHS}}

@torch.inference_mode()
def neural_file(model,raw,device,completion=True,batch=64,measure_latency=False):
    result=initial();chunks=list(blocks(raw))
    for lo in range(0,len(chunks),batch):
        bunch=chunks[lo:lo+batch]
        x=torch.zeros((len(bunch),127),dtype=torch.long,device=device)
        for i,(prefix,gold) in enumerate(bunch):x[i,:len(prefix)]=torch.tensor(prefix,device=device)
        logits=model(x).double()
        allprob=torch.softmax(logits,-1).cpu().numpy()
        for i,(_,gold) in enumerate(bunch):add_predictions(result,allprob[i,63:63+len(gold)],gold)
    finish(result)
    if not completion:return result
    positions=offers(raw);rows=[];times=[]
    for lo in range(0,len(positions),batch):
        pos=positions[lo:lo+batch]
        x=torch.tensor([list(raw[p-128:p]) for p in pos],dtype=torch.long,device=device)
        generated=[];confidence=None
        if device.startswith('cuda'):torch.cuda.synchronize()
        start=time.perf_counter()
        for step in range(32):
            logits=model(x)[:,-1]
            prob=torch.softmax(logits,-1)
            if step==0:confidence=prob.max(-1).values.cpu().tolist()
            nxt=logits.argmax(-1);generated.append(nxt)
            x=torch.cat((x,nxt[:,None]),dim=1)[:,-128:]
            if device.startswith('cuda'):torch.cuda.synchronize()
            if step+1 in LENGTHS:
                # Batch elapsed is user-visible latency for a batch of this size,
                # never mislabel throughput division as single-user latency.
                times.append({'batch':len(pos),'length':step+1,
                              'batch_elapsed_ms':(time.perf_counter()-start)*1000})
        predicted=torch.stack(generated,dim=1).cpu().tolist()
        for p,g,c in zip(pos,predicted,confidence):
            row=suggestion_row(p,g,raw[p:p+32]);row['first_byte_confidence']=c;rows.append(row)
    result.update(offers=rows,batched_suggestion_timings=times)
    # Separate actual batch1 timing on16 fixed opportunities,8 warmup excluded.
    # Includes repeated full-window recomputation; not a streaming kernel claim.
    if not measure_latency:return result
    singles={str(n):[] for n in LENGTHS}
    for j,p in enumerate(positions[:24]):
        x=torch.tensor([list(raw[p-128:p])],dtype=torch.long,device=device)
        if device.startswith('cuda'):torch.cuda.synchronize()
        started=time.perf_counter()
        for k in range(32):
            nxt=model(x)[:,-1].argmax(-1)
            x=torch.cat((x,nxt[:,None]),1)[:,-128:]
            if device.startswith('cuda'):torch.cuda.synchronize()
            if j>=8 and k+1 in LENGTHS:singles[str(k+1)].append((time.perf_counter()-started)*1000)
    result['batch1_suggestion_ms']={n:{'p50':float(np.median(v)),'p95':float(np.quantile(v,.95)),
                                          'samples':len(v)} for n,v in singles.items()}
    return result

class Counts:
    """Fixed FIT-only P3 or interpolated Witten-Bell order6, no online updates."""
    def __init__(self,docs,order):
        self.order=order;self.counts=[defaultdict(Counter) for _ in range(order+1)]
        uni=Counter()
        for raw in docs:
            uni.update(raw)
            for t in range(1,len(raw)):
                for n in ([3] if order==3 and t>=3 else [] if order==3 else range(1,min(order,t)+1)):
                    self.counts[n][raw[t-n:t]][raw[t]]+=1
        self.prior=np.array([uni[i]+1 for i in range(256)],dtype=np.float64)
        self.prior/=self.prior.sum()
        # Original P3 counted trigram contexts only from t3 onward; exact.
        self.logical_bytes=sum(n+6*len(v) for n,ct in enumerate(self.counts) for v in ct.values())+2048
    @lru_cache(maxsize=8192)
    def probs(self,ctx):
        if not ctx:return self.prior
        rows=self.counts[len(ctx)].get(ctx,{})
        if self.order==3:
            count=sum(rows.values());p=self.prior*4
            p=p.copy()
            for b,n in rows.items():p[b]+=n
            return p/(count+4)
        lower=self.probs(ctx[1:])
        if not rows:return lower
        count=sum(rows.values());types=len(rows)
        p=lower*types
        for b,n in rows.items():p[b]+=n
        return p/(count+types)
    def distribution(self,ctx):return self.probs(bytes(ctx[-self.order:]))
    def score(self,raw):
        result=initial()
        for base in range(64,len(raw),1024):
            stop=min(base+1024,len(raw))
            prob=np.stack([self.distribution(raw[max(0,i-self.order):i]) for i in range(base,stop)])
            add_predictions(result,prob,list(raw[base:stop]))
        finish(result);rows=[];timings={str(n):[] for n in LENGTHS}
        for j,p in enumerate(offers(raw)):
            context=raw[p-128:p];g=[];start=time.perf_counter()
            for k in range(32):
                g.append(int(self.distribution(context+bytes(g)).argmax()))
                if k+1 in LENGTHS:timings[str(k+1)].append((time.perf_counter()-start)*1000)
            rows.append(suggestion_row(p,g,raw[p:p+32]))
        result.update(offers=rows,logical_count_store_estimate_bytes=self.logical_bytes,
                      batch1_suggestion_ms={n:{'p50':float(np.median(v)),'p95':float(np.quantile(v,.95)),
                                                 'samples':len(v)} for n,v in timings.items()})
        return result
