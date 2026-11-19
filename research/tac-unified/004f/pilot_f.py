"""004F: bounded curiosity mechanism diagnostic. Frozen no-fit, unknown hidden target."""
import os,json,time
from pathlib import Path
import numpy as np
import prior004 as old
import prior004cd as cd
import prior004e as e
B,N=old.B,old.N
CAP,GROUP,CASES=4096,64,32
ARMS={'passive':0,'systematic_16':16,'random_4':4,'systematic_4':4,'systematic_all':1}
def schedule(tab,model,seed,name):
    # No hidden labels. Common passive fixture across arms.
    rng=np.random.default_rng(seed)
    targets=np.array(sorted(model.ins))
    bs=rng.choice(targets,CAP); states=rng.integers(N,size=(CAP,B))
    ds=np.flatnonzero(tab[1]!=tab[2])[1:]
    grid=np.array([(int(b),int(v)) for b in targets for v in ds])
    order=np.random.default_rng(seed+1).permutation(len(grid))
    random=np.random.default_rng(seed+2)
    stride=ARMS[name]; active=np.zeros(CAP,bool)
    if stride:
        ix=np.arange(stride-1,CAP,stride)
        if name=='random_4': chosen=random.integers(len(grid),size=len(ix))
        else:
            ix=ix[:len(grid)];chosen=order[:len(ix)]
        for t,(b,v) in zip(ix,grid[chosen]):
            states[t]|=1; states[t,model.ins[int(b)]]&=~1
            states[t,b]=v;bs[t]=b;active[t]=True
    if name=='systematic_all':
        length=min(CAP,len(grid));states=states[:length];bs=bs[:length];active=active[:length]
    return states,bs,active
def monitor(world,model,states,bs,active):
    start=time.perf_counter_ns();found=None;spent=0;before=world.queries
    for lo in range(0,len(bs),GROUP):
        hi=min(lo+GROUP,len(bs));y=np.empty_like(states[lo:hi]);p=y.copy()
        for b in np.unique(bs[lo:hi]):
            ix=np.flatnonzero(bs[lo:hi]==b)
            x=states[lo+ix]; y[ix]=world.step(x,int(b),1);p[ix]=model.step(x,int(b),1)
        spent=hi
        err=np.flatnonzero(np.any(y!=p,axis=1))
        if len(err):
            k=int(err[0]);found=(lo+k,states[lo+k].copy(),int(bs[lo+k]),y[k].copy());break
    assert world.queries-before==spent
    row={'detected':found is not None,'delay':int(found[0]+1) if found else CAP,'spent':spent,'active':int(active[:spent].sum()),'passive':int(spent-active[:spent].sum()),'monitor_ms':(time.perf_counter_ns()-start)/1e6}
    return found,row
def run_case(tab,truth,seed,case,clean=False):
    r=np.random.default_rng(seed+290000+case*101+int(clean)*90000)
    order=r.permutation(B);edges=list(zip(map(int,order[::2]),map(int,order[1::2])))
    p,b=edges[int(r.integers(len(edges)))];ds=np.flatnonzero(tab[1]!=tab[2])[1:]
    h=int(r.choice([i for i in range(B) if i not in (p,b)]));v=int(r.choice(ds));hidden=(h,b,v)
    def make():return old.World(truth,edges) if clean else cd.DormantWorld(truth,edges,hidden)
    initial=make();got=cd.targeted(initial,tab);pairs=old.matching(*got[:2]);assert pairs is not None
    model=old.World(tab,pairs);rows={}
    # Independent validation schedule shared across arms; trigger-enrichment only for scoring.
    vr=np.random.default_rng(seed+390000+case*101+int(clean)*90000)
    check=vr.integers(N,size=(256,B));check[:128,b]=v
    for arm in ARMS:
        # Equal passive stream / active candidate order for matched arms.
        states,bs,active=schedule(tab,model,seed+490000+case*101+int(clean)*90000,arm)
        world=make();found,row=monitor(world,model,states,bs,active)
        row.update(repaired=False,repair_queries=0,localization=0,repair_ms=0.0,checks=0,after_errors=0,correct_parent=False,initial_queries=initial.queries)
        if found and not clean:
            _,s,target,y=found;before=world.queries;t=time.perf_counter_ns()
            rule,probes,reason=e.repair(world,model,s,target,1,y)
            row.update(repair_queries=world.queries-before,localization=probes,repair_ms=(time.perf_counter_ns()-t)/1e6,reason=reason)
            if rule is not None:
                revised=e.Revised(tab,pairs,rule)
                after=world.step(check,b,1);pred=revised.step(check,b,1)
                row.update(correct_parent=bool(rule[0]==h and rule[1]==b),checks=len(check),after_errors=int(np.any(after!=pred,axis=1).sum()))
                row['repaired']=row['correct_parent'] and row['after_errors']==0
        rows[arm]=row
    return {'hidden':list(hidden),'clean':clean,'arms':rows,'grid_size':len(model.ins)*len(ds)}
def main():
    seed=int(os.environ['TRAINLAB_SEED']);out=Path(os.environ['TRAINLAB_OUTPUT_DIR']);out.mkdir(parents=True,exist_ok=True)
    _,_,tab,truth=old.load(seed)
    cases=[run_case(tab,truth,seed,i) for i in range(CASES)]
    clean=[run_case(tab,truth,seed,i,True) for i in range(4)]
    summary={}
    for arm in ARMS:
        a=[c['arms'][arm] for c in cases]
        summary[arm]={'n':len(a),'detected':sum(x['detected'] for x in a),'repaired':sum(x['repaired'] for x in a),'false_alerts':sum(c['arms'][arm]['detected'] for c in clean)}
        for key in ['delay','spent','active','passive','repair_queries','localization','repair_ms','monitor_ms','checks','after_errors','initial_queries']:
            summary[arm][key+'_sum']=sum(x[key] for x in a)
    s=summary
    gates={'quarter':s['systematic_4']['detected']>=s['passive']['detected'] and s['systematic_4']['repaired']==s['systematic_4']['detected'] and s['systematic_4']['after_errors_sum']==0,
           'full':s['systematic_all']['repaired']==CASES and s['systematic_all']['after_errors_sum']==0,
           'clean':all(x['false_alerts']==0 for x in s.values())}
    result={'seed':seed,'qualification':'NOT_QUALIFIED','summary':summary,'gates':gates,'cases':cases,'clean':clean}
    metrics={'passed_gates':sum(gates.values()),'passive_detected':s['passive']['detected'],'quarter_detected':s['systematic_4']['detected'],'full_detected':s['systematic_all']['detected'],'quarter_active':s['systematic_4']['active_sum']}
    for name,obj in [('results',result),('metrics',metrics)]: (out/(name+'.json')).write_text(json.dumps(obj,indent=2,sort_keys=True)+'\n')
    print(seed,metrics,gates,flush=True)
if __name__=='__main__':main()
