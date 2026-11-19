"""005A1 no-fit, actual native CSL and trusted-process generation manager."""
import os,sys,json,copy,threading,statistics,time,hashlib,importlib
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
import prior004 as old
import prior005a as p
from native_model import selected_native_byte
ROOT=Path(__file__).resolve().parent
class Generation:
    def __init__(self,g,model,bank,tab,prev=None):
        self.g=g
        self._model=copy.deepcopy(model).eval().requires_grad_(False)
        self._bank=bank.detach().clone()
        self._tab=np.array(tab,copy=True);self._tab.flags.writeable=False
        self._lock=threading.Lock()
        t=time.perf_counter_ns();self.pin=p.binding(self._model,self._bank,self._tab);self.hash_ms=(time.perf_counter_ns()-t)/1e6
        self.cache={};self.inherited=0
        for a,b in [(0,1),(2,3)]:
            key=(g,a,b)
            if prev is not None and np.array_equal(prev._tab[[a,b]],self._tab[[a,b]]):
                self.cache[key]=prev.cache[(prev.g,a,b)];self.inherited+=1
            else:
                val=self._tab[b,self._tab[a,np.arange(256)]];val.flags.writeable=False;self.cache[key]=val
    def encode(self,prompt,state=None):
        with self._lock,torch.inference_mode():
            z,st,logits=p.features(self._model,prompt,state)
            assert not logits
            slot=int((self._bank@F.normalize(z,dim=0)).argmax())
        return slot,st
    def check(self,token):
        if token!=self.g:raise ValueError('wrong generation')
    def execute(self,token,slot,states):
        self.check(token);return self._tab[slot,states]
    def anchors(self):return [self.encode(x)[0] for x in p.PROMPTS]
class Lease:
    def __init__(self,g):self._gen=g;self._state=None;self._closed=False
    def request(self,prompt,states,carry=False):
        if self._closed:raise ValueError('closed')
        slot,st=self._gen.encode(prompt,self._state if carry else None)
        self._state=st if carry else None
        return self._gen.g,slot,self._gen.execute(self._gen.g,slot,states)
    def close(self):self._state=None;self._gen=None;self._closed=True
class Manager:
    def __init__(self,g):self._lock=threading.Lock();self._active=g
    def acquire(self):
        with self._lock:return Lease(self._active)
    def promote(self,candidate):
        # candidate prepared and tested before pointer swap; reject race/ABA
        if candidate.anchors()!=list(range(6)):return False
        with self._lock:
            if candidate.g!=self._active.g+1:return False
            self._active=candidate;return True
def outcome(tab,truth,seed):
    r=np.random.default_rng(seed+590000);a=np.repeat(np.arange(6),256);s=np.tile(np.arange(256),6);y=truth[a,s]
    mask=tab[:,s].T==y[:,None];s2=r.integers(256,size=len(a));m2=mask&(tab[:,s2].T==truth[a,s2,None])
    # independently computed match oracle, two independent states same action supplied
    def score(m):
        c=m.sum(1);return dict(n=len(c),contains=int(m[np.arange(len(a)),a].sum()),unique=int((c==1).sum()),wrong_unique=int(((c==1)&~m[np.arange(len(a)),a]).sum()),hist=np.bincount(c,minlength=7).tolist())
    x=score(mask);x2=score(m2)
    bad=np.array([next(i for i in range(256) if i not in row) for row in tab[:,s].T]);reject=int((~(tab[:,s].T==bad[:,None]).any(1)).sum())
    # Changed true action 1 now acts exactly like known action 2
    ss=np.flatnonzero(tab[1]!=tab[2]);alias=tab[:,ss].T==tab[2,ss,None];alias_unique=int((alias.sum(1)==1).sum())
    z=torch.tensor(r.normal(size=mask.shape),dtype=torch.float64,requires_grad=True);m=torch.from_numpy(mask)
    loss=(torch.logsumexp(z,1)-torch.logsumexp(z.masked_fill(~m,-torch.inf),1)).mean()
    oracle=-(torch.softmax(z.detach(),1)*m).sum(1).log().mean()
    loss.backward();diff=float(abs(loss.detach()-oracle));finite=bool(torch.isfinite(z.grad).all())
    duplicates=[(i,j) for i in range(6) for j in range(i+1,6) if np.array_equal(tab[i],tab[j])]
    return dict(one=x,two=x2,out_of_model_rejected=reject,alias_n=len(ss),alias_false_unique=alias_unique,duplicates=duplicates,loss=float(loss.detach()),loss_difference=diff,finite_grad=finite,gate=x['contains']==len(a) and x2['contains']==len(a) and x['wrong_unique']==x2['wrong_unique']==0 and x2['unique']>=x['unique'] and reject==len(a) and finite and diff<=1e-10)
def main():
    seed=int(os.environ['TRAINLAB_SEED']);out=Path(os.environ['TRAINLAB_OUTPUT_DIR']);out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.manual_seed(seed)
    pins=json.loads((ROOT/'source-hashes.json').read_text());countpins=0
    for name,h in pins.items():
        if name.startswith('tac_osm.'):
            assert hashlib.sha256(Path(importlib.import_module(name).__file__).read_bytes()).hexdigest()==h
            countpins+=1
    _,_,tab,truth=old.load(seed);m=selected_native_byte().eval();params=sum(x.numel() for x in m.parameters())
    with torch.inference_mode():bank=F.normalize(torch.stack([p.features(m,x)[0] for x in p.PROMPTS]),dim=1)
    g0=Generation(0,m,bank,tab);manager=Manager(g0)
    # mutate donor before building new version; g0 must remain unaffected
    with torch.no_grad():
        rows=sorted(set(b for x in p.PROMPTS for b in x));m.core.embedding.weight[rows,:32]+=0.01
    shifted=Generation(1,m,bank,tab,g0);states=np.arange(256);original_pin=g0.pin
    old_lease=manager.acquire();ready=threading.Event();resume=threading.Event();record={};errors=[]
    def inflight():
        try:
            x=old_lease.request(p.PROMPTS[0][:11],states,True);ready.set()
            if not resume.wait(20):raise RuntimeError('barrier timeout')
            record['old']=old_lease.request(p.PROMPTS[0][11:],states,True);old_lease.close()
        except BaseException as exc:errors.append(str(exc));ready.set()
    th=threading.Thread(target=inflight);th.start();assert ready.wait(20)
    promoted=manager.promote(shifted);new_lease=manager.acquire();record['new']=new_lease.request(p.PROMPTS[0],states);resume.set();th.join(20);assert not th.is_alive()
    reference=g0.encode(p.PROMPTS[0])[0]
    inflight_ok=not errors and record['old'][0]==0 and record['old'][1]==reference and np.array_equal(record['old'][2],truth[reference]) and record['new'][0]==1 and record['new'][1]==0
    stale=False
    try:shifted.execute(0,0,states)
    except ValueError:stale=True
    closed=False
    try:old_lease.request(p.PROMPTS[0],states)
    except ValueError:closed=True
    # donor mutation must not affect snapshot; table donor mutation also tested
    donor=tab.copy();donor[0,0]^=1
    changed=Generation(2,m,bank,donor,shifted);saved=changed._tab.copy();donor[:]=0
    with torch.no_grad():m.core.embedding.weight.add_(0.1)
    isolated=p.binding(g0._model,g0._bank,g0._tab)==original_pin and p.binding(shifted._model,shifted._bank,shifted._tab)==shifted.pin and np.array_equal(changed._tab,saved)
    copied=(shifted.cache[(1,0,1)] is g0.cache[(0,0,1)] and shifted.cache[(1,2,3)] is g0.cache[(0,2,3)])
    selective=(changed.cache[(2,0,1)] is not shifted.cache[(1,0,1)] and changed.cache[(2,2,3)] is shifted.cache[(1,2,3)] and np.array_equal(changed.cache[(2,0,1)],saved[1,saved[0,states]]) and np.array_equal(g0.cache[(0,0,1)],truth[1,truth[0,states]]))
    incompatible=Generation(2,shifted._model,torch.roll(bank,1,0),tab,shifted);rejected=not manager.promote(incompatible)
    timing={'hash_ms':[],'dispatch_us':[],'check_ns':[]}
    for _ in range(7):
        t=time.perf_counter_ns();assert p.binding(g0._model,g0._bank,g0._tab)==g0.pin;timing['hash_ms'].append((time.perf_counter_ns()-t)/1e6)
        t=time.perf_counter_ns()
        for i in range(2000):g0.execute(0,0,states)
        timing['dispatch_us'].append((time.perf_counter_ns()-t)/2e6)
        t=time.perf_counter_ns()
        for i in range(2000):g0.check(0)
        timing['check_ns'].append((time.perf_counter_ns()-t)/2000)
    o=outcome(tab,truth,seed)
    gates={'pins':countpins==6 and params==10103943,'promote':promoted,'inflight':inflight_ok,'stale':stale and closed,'isolation':isolated,'cache':copied and selective,'incompatible':rejected,'outcome':o['gate']}
    result=dict(seed=seed,qualification='NOT_QUALIFIED',parameters=params,gates=gates,timing=timing,medians={k:statistics.median(v) for k,v in timing.items()},outcome=o,anchors=shifted.anchors(),inherited=shifted.inherited,selective_inherited=changed.inherited,errors=errors,hash_pins=[g0.pin,shifted.pin,changed.pin],initial_hash_ms=[g0.hash_ms,shifted.hash_ms,changed.hash_ms])
    metrics={'passed_gates':sum(gates.values()),'unique_coverage':o['one']['unique']/o['one']['n'],'two_coverage':o['two']['unique']/o['two']['n'],'dispatch_us':result['medians']['dispatch_us']}
    for name,obj in [('results',result),('metrics',metrics)]:(out/(name+'.json')).write_text(json.dumps(obj,indent=2,sort_keys=True)+'\n')
    print(seed,metrics,gates,flush=True)
if __name__=='__main__':main()
