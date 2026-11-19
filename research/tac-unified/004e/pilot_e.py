"""004E no-fit diagnostic. No hidden graph or gold action passed to repair/decoder."""
import json, os, time
from pathlib import Path
import numpy as np
import prior004 as old
import prior004cd as cd
B,N,K,T,S=old.B,old.N,old.K,old.T,old.SESSIONS

class AndWorld(cd.DormantWorld):
    def __init__(self,tab,edges,hidden,gate):
        super().__init__(tab,edges,hidden); self.gate=gate
    def step(self,s,b,a):
        x=old.World.step(self,s,b,a); i,j,v=self.hidden
        if b==j and a==1:
            s=np.asarray(s); on=(s[...,i]&1)&(s[...,self.gate]&1)&(s[...,j]==v)
            x[...,j]=np.where(on,self.table[2,s[...,j]],x[...,j])
        return x

class Revised(old.World):
    def __init__(self,tab,edges,rule):
        super().__init__(tab,edges);self.rule=rule
    def step(self,s,b,a):
        x=super().step(s,b,a);i,j,mask=self.rule
        if b==j and a==1:
            s=np.asarray(s);on=(s[...,i]&1)&mask[s[...,j]]
            x[...,j]=np.where(on,self.table[2,s[...,j]],x[...,j])
        return x

def repair(world,model,s,b,a,actual):
    """Candidate single-cause isolation. All calls are counted simulator interventions."""
    if a!=1 or b not in model.ins: return None,0,'unsupported action'
    p=model.ins[b];active=[i for i in range(B) if i not in (p,b) and s[i]&1]
    base=s.copy();base[active]&=~1
    probes=0
    anchor_residual=actual ^ model.step(s,b,a)
    def run(sub):
        nonlocal probes
        x=base.copy();x[sub]|=1; probes+=1
        return world.step(x,b,a) ^ model.step(x,b,a)
    if not active or np.any(run([])):
        return None,probes,'baseline inconsistent'
    # keep a half of active bits; if neither reproduces effect, reject interaction
    while len(active)>1:
        half=len(active)//2; left,right=active[:half],active[half:]
        if np.array_equal(run(left),anchor_residual):active=left
        elif np.array_equal(run(right),anchor_residual):active=right
        else:return None,probes,'no isolated half reproduces residual'
    if not np.array_equal(run(active),anchor_residual):return None,probes,'singleton fails'
    i=active[0]
    # verify full local gate; all other unknown bits off, known parent off
    x=np.repeat(base[None,:],N,0); x[:,b]=np.arange(N);x[:,p]&=~1;x[:,i]&=~1
    lo=world.step(x,b,a)[:,b];x[:,i]|=1;hi=world.step(x,b,a)[:,b]
    if not np.array_equal(lo,model.table[1]) or not np.all((hi==lo)|(hi==model.table[2])):
        return None,probes,'unrecognized sweep'
    mask=hi!=lo
    if not mask[s[b]]:return None,probes,'anchor not covered'
    return (i,b,mask),probes,None

def run1(tab,truth,seed):
    r=np.random.default_rng(seed+190000);rows=[]
    for case in range(16):
        order=r.permutation(B);edges=list(zip(map(int,order[::2]),map(int,order[1::2])))
        p,b=edges[0]; ds=np.flatnonzero(tab[1]!=tab[2]);v=int(r.choice(ds[1:]))
        h=int(r.choice([i for i in range(B) if i not in (p,b)]));hidden=(h,b,v)
        world=cd.DormantWorld(truth,edges,hidden)
        # actual targeted method, including probes. Construct model only from recovered edges.
        start=time.perf_counter_ns(); got=cd.targeted(world,tab); pairs=old.matching(*got[:2]);initial_q=world.queries
        if pairs is None: raise ValueError('initial graph rejected')
        model=old.World(tab,pairs); found=None
        for delay in range(1,2049):
            s=r.integers(N,size=B);y=world.step(s,b,1)
            if not np.array_equal(y,model.step(s,b,1)):found=(s,y);break
        rec={'initial_queries':initial_q,'delay':delay,'detected':found is not None,'hidden':hidden}
        if found:
            s,y=found;before=world.queries;t=time.perf_counter_ns();rule,probes,reason=repair(world,model,s,b,1,y)
            rec.update(repair_queries=world.queries-before,localization_probes=probes,repair_ms=(time.perf_counter_ns()-t)/1e6,reason=reason,invalidated=True)
            if rule is not None:
                revised=Revised(tab,pairs,rule);i,j,mask=rule; rec.update(recovered=int(i),correct_parent=bool(i==h),gate_states=np.flatnonzero(mask).tolist());before_err=after_err=0
                for k in range(256):
                    z=r.integers(N,size=B)
                    if k<128:z[b]=int(r.choice(np.flatnonzero(mask)))
                    actual=world.step(z,b,1);before_err+=int(not np.array_equal(actual,model.step(z,b,1)));after_err+=int(not np.array_equal(actual,revised.step(z,b,1)))
                rec.update(before_errors=before_err,after_errors=after_err,checks=256)
        rec['elapsed_ms']=(time.perf_counter_ns()-start)/1e6;rows.append(rec)
    # Additional AND-gated interaction test: supplied triggering snapshot, not supplied hidden parent to repair.
    g=next(i for i in range(B) if i not in (p,b,h));aw=AndWorld(truth,edges,hidden,g)
    s=np.zeros(B,int);s[b]=v;s[h]=s[g]=1;y=aw.step(s,b,1)
    assert not np.array_equal(y,model.step(s,b,1))
    rule,probes,reason=repair(aw,model,s,b,1,y)
    success=[x for x in rows if x.get('correct_parent') and x.get('after_errors')==0]
    gate=len(success)>=15 and all(x['initial_queries']+x['repair_queries']<.1*262208 for x in success) and rule is None
    return {'rows':rows,'repaired':len(success),'and_rejected':rule is None,'and_reason':reason,'and_probes':probes,'gate':bool(gate)}

def run2(code,tab,truth,seed):
    r=np.random.default_rng(seed+191000);a=r.integers(K,size=(T,S));b=r.integers(16,size=a.shape);state=r.integers(N,size=(S,16));rr=np.arange(S)
    pre=np.empty_like(a);post=np.empty_like(a)
    for t in range(T):
        pre[t]=state[rr,b[t]];post[t]=truth[a[t],pre[t]];state[rr,b[t]]=post[t]
    # actual pre/post states are an explicit diagnostic-only observed modality.
    cue=old.flip(r,code[a.ravel()],8); gold=a.ravel();start=time.perf_counter_ns();d=old.distances(cue,code);pred=d.argmin(1);cue_ms=(time.perf_counter_ns()-start)/1e6
    start=time.perf_counter_ns();matches=tab[:,pre.ravel()].T==post.ravel()[:,None];count=matches.sum(1); consistent=matches[np.arange(len(pred)),pred];corrected=np.where(count>0,np.where(matches,d,32767).argmin(1),-1);unique=np.where(count==1,matches.argmax(1),-1);both_ms=cue_ms+(time.perf_counter_ns()-start)/1e6
    def score(p):
        err=p!=gold;return {'correct':int((~err).sum()),'exact_chains':int((~err.reshape(T,S)).all(0).sum()),'accepted':int((p>=0).sum()),'wrong_accepted':int((err&(p>=0)).sum())}
    # alias: actual action1 with changed transition equal to known action2. Verified true cue=1.
    ds=np.flatnonzero(tab[1]!=tab[2]);q=r.choice(ds,4096);alias_y=tab[2,q];m=tab[:,q].T==alias_y[:,None];cnt=m.sum(1)
    # correct known action absent; unique-match would incorrectly label a structural change as cue error.
    alias={'n':len(q),'wrong_perception_label':int((cnt==1).sum()),'ambiguous':int((cnt>1).sum()),'no_match':int((cnt==0).sum())}
    # independent out-of-model and repeated wrong cues at same slot: no causal streak rule.
    x=r.integers(N,size=4096);possible=tab[:,x].T;bad=np.array([next(v for v in range(N) if v not in row) for row in possible]);out_rej=int((~(possible==bad[:,None]).any(1)).sum())
    streak_n=1024;streak_state=np.resize(ds,streak_n);streak_post=tab[1,streak_state];streak_guess=np.full(streak_n,2);streak_miss=(tab[streak_guess,streak_state]!=streak_post);streak_false=int((streak_miss[1:]&streak_miss[:-1]).sum())
    metrics={'n':len(gold),'chains':S,'cue':score(pred),'combined':score(corrected),'unique':score(unique),'ambiguous':int((count>1).sum()),'mismatches':int((~consistent).sum()),'cue_ms':cue_ms,'combined_ms':both_ms,'alias':alias,'out_of_model_rejected':out_rej,'out_of_model_n':len(x),'repeated_perception_false_structural_streaks':streak_false}
    metrics['gate']=bool(metrics['combined']['correct']>=metrics['cue']['correct'] and metrics['unique']['wrong_accepted']==0 and out_rej==len(x));return metrics

def main():
    seed=int(os.environ['TRAINLAB_SEED']);out=Path(os.environ['TRAINLAB_OUTPUT_DIR']);out.mkdir(parents=True,exist_ok=True)
    _,code,tab,truth=old.load(seed);e1=run1(tab,truth,seed);e2=run2(code,tab,truth,seed);gates={'revision':e1['gate'],'guard':e2['gate']};result={'seed':seed,'qualification':'NOT_QUALIFIED','e1':e1,'e2':e2,'gates':gates}
    metrics={'passed_gates':sum(gates.values()),'repaired':e1['repaired'],'guard_accuracy':e2['combined']['correct']/e2['n'],'cue_accuracy':e2['cue']['correct']/e2['n'],'unique_coverage':e2['unique']['accepted']/e2['n']}
    for name,obj in [('results',result),('metrics',metrics)]: (out/(name+'.json')).write_text(json.dumps(obj,indent=2,sort_keys=True)+'\n')
    print(seed,metrics,gates,flush=True)
if __name__=='__main__':main()
