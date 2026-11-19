"""005A2 diagnostic, no fit."""
import os,json,time,statistics,hashlib,importlib
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
import prior004 as old
import prior005a as p
import prior005a1 as g
from native_model import selected_native_byte
ROOT=Path(__file__).resolve().parent
def active(tab,truth,seed):
    rng=np.random.default_rng(seed+690000);n=1536
    a=np.repeat(np.arange(6),256);s=np.tile(np.arange(256),6);y=truth[a,s]
    mask=tab[:,s].T==y[:,None]
    r=rng.integers(255,size=n);r+=r>=s
    random=mask&(tab[:,r].T==truth[a,r][:,None])
    # Enumerate pairwise equality for possible candidate subsets, all states.
    eq=tab[:,None,:]==tab[None,:,:];scores={}
    for bits in range(1,64):
        ids=[j for j in range(6) if bits>>j&1]
        scores[bits]=eq[np.ix_(ids,ids,np.arange(256))].sum((0,1))
    selected=mask.copy();two=mask.copy();reads=np.ones(n,int);chosen=[];prep_start=time.perf_counter_ns()
    for i in range(n):
        m=mask[i].copy();seen=[int(s[i])];path=[]
        for k in range(5):
            if m.sum()==1:break
            bits=sum(1<<j for j in np.flatnonzero(m));cost=scores[bits].copy();cost[seen]=100
            q=int(cost.argmin());seen.append(q);path.append(q);m&=tab[:,q]==truth[a[i],q];reads[i]+=1
            if k==0:two[i]=m
        selected[i]=m;chosen.append(path)
    choose_ms=(time.perf_counter_ns()-prep_start)/1e6
    # unique original rows need no second read; random control nonetheless offered a second observation
    def stat(m):
        c=m.sum(1);return dict(unique=int((c==1).sum()),contains=int(m[np.arange(n),a].sum()),wrong_unique=int(((c==1)&~m[np.arange(n),a]).sum()),hist=np.bincount(c,minlength=7).tolist())
    st=[stat(m) for m in (mask,random,two,selected)]
    dup=sum(np.array_equal(tab[i],tab[j]) for i in range(6) for j in range(i+1,6))
    return dict(n=n,one=st[0],random_distinct=st[1],active_two=st[2],active_six=st[3],reads_hist=np.bincount(reads,minlength=7).tolist(),mean_reads=float(reads.mean()),duplicates=dup,selection_ms=choose_ms,paths=chosen,gate=all(x['contains']==n and x['wrong_unique']==0 for x in st) and st[2]['unique']>=st[1]['unique'] and (dup>0 or st[3]['unique']==n))
def alias(tab,truth,seed):
    r=np.random.default_rng(seed+691000);n=2048;s=r.choice(np.flatnonzero(tab[1]!=tab[2]),n);a=r.integers(6,size=n);a[r.random(n)<.5]=1
    y0=truth[a,s];y=y0.copy();shifted=(np.arange(n)>=1024)&(a==1);y[shifted]=truth[2,s[shifted]]
    def arms(y):
        m=tab[:,s].T==y[:,None];bad=~m[np.arange(n),a];single=m.sum(1)==1;out={}
        for name,stride in [('ungated',0),('compatible_only',0),('audit64',64),('strict',1)]:
            admit=np.zeros(n,bool);frozen=False;first=-1
            for t in range(n):
                if stride and t%stride==0 and bad[t] and not frozen:frozen=True;first=t
                admit[t]=not frozen and bool(m[t].any())
            # only admitted rows contribute, avoid inf*0.
            z=torch.zeros((n,6),dtype=torch.float64,requires_grad=True);ix=np.flatnonzero(admit);m2=torch.from_numpy(m[ix]);zz=z[ix]
            loss=(torch.logsumexp(zz,1)-torch.logsumexp(zz.masked_fill(~m2,-torch.inf),1)).sum()
            loss.backward();zero=bool((z.grad[~admit]==0).all())
            out[name]=dict(admitted=len(ix),bad=int((bad&admit).sum()),false_unique=int((bad&admit&single).sum()),first_freeze=first,rejected_zero_grad=zero,finite=bool(torch.isfinite(z.grad).all()))
        return out
    clean=arms(y0);shift=arms(y)
    return dict(n=n,shifted=int(shifted.sum()),clean=clean,shift=shift,gate=shift['strict']['bad']==0 and shift['strict']['false_unique']==0 and all(x['admitted']==n and x['first_freeze']==-1 for x in clean.values()) and all(x['rejected_zero_grad'] and x['finite'] for a0 in (clean,shift) for x in a0.values()))
def latency(tab,truth,seed):
    torch.manual_seed(seed);m=selected_native_byte().eval();params=sum(x.numel() for x in m.parameters())
    with torch.inference_mode():bank=F.normalize(torch.stack([p.features(m,x)[0] for x in p.PROMPTS]),dim=1)
    t=time.perf_counter_ns();gen=g.Generation(0,m,bank,tab);mgr=g.Manager(gen);prep_ms=(time.perf_counter_ns()-t)/1e6
    def request(i,hashed):
        lease=mgr.acquire()
        if hashed:assert p.binding(gen._model,gen._bank,gen._tab)==gen.pin
        token,slot,answer=lease.request(p.PROMPTS[i],np.array([i*37]));lease.close()
        return token==0 and slot==i and int(answer[0])==int(truth[i,i*37])
    for _ in range(2):
        for i in range(6):
            for hashed in (False,True):assert request(i,hashed)
    timing={'lease_ms':[],'hash_ms':[],'dispatch_us':[],'enumerate_us':[],'check_ns':[]};correct=0
    states=np.array([113])
    for j in range(7):
        for hashed in (bool(j%2),not bool(j%2)):
            for i in range(6):
                t=time.perf_counter_ns();ok=request(i,hashed);dt=(time.perf_counter_ns()-t)/1e6;correct+=int(ok);timing['hash_ms' if hashed else 'lease_ms'].append(dt)
        t=time.perf_counter_ns()
        for _ in range(2000):gen.execute(0,0,states)
        timing['dispatch_us'].append((time.perf_counter_ns()-t)/2e6)
        t=time.perf_counter_ns()
        for _ in range(2000):tab[:,113]
        timing['enumerate_us'].append((time.perf_counter_ns()-t)/2e6)
        t=time.perf_counter_ns()
        for _ in range(2000):gen.check(0)
        timing['check_ns'].append((time.perf_counter_ns()-t)/2000)
    return dict(params=params,correct=correct,n=84,prep_ms=prep_ms,initial_hash_ms=gen.hash_ms,timing=timing,median={k:statistics.median(v) for k,v in timing.items()},gate=correct==84 and params==10103943)
def main():
    seed=int(os.environ['TRAINLAB_SEED']);out=Path(os.environ['TRAINLAB_OUTPUT_DIR']);out.mkdir(parents=True,exist_ok=True);torch.set_num_threads(1);torch.set_num_interop_threads(1)
    pins=json.loads((ROOT/'source-hashes.json').read_text());n=0
    for name,h in pins.items():
        if name.startswith('tac_osm.'):
            assert hashlib.sha256(Path(importlib.import_module(name).__file__).read_bytes()).hexdigest()==h;n+=1
    _,_,tab,truth=old.load(seed);a=active(tab,truth,seed);b=alias(tab,truth,seed);c=latency(tab,truth,seed)
    gates=dict(active=a['gate'],alias=b['gate'],latency=c['gate'] and n==6)
    result=dict(seed=seed,qualification='NOT_QUALIFIED',active=a,alias=b,latency=c,gates=gates)
    metrics=dict(passed_gates=sum(gates.values()),active_two=a['active_two']['unique']/1536,active_six=a['active_six']['unique']/1536,strict_bad=b['shift']['strict']['bad'],lease_ms=c['median']['lease_ms'])
    for name,obj in [('results',result),('metrics',metrics)]:(out/(name+'.json')).write_text(json.dumps(obj,indent=2,sort_keys=True)+'\n')
    print(seed,metrics,gates,flush=True)
if __name__=='__main__':main()
