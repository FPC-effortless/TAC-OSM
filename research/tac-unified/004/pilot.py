"""No-fit mechanism diagnostic for TAC-UNIFIED-004A/B; invoked by TrainLab."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
import collections
import hashlib
import importlib.util
import itertools
import json
import math
from pathlib import Path
import tempfile
import time
import zipfile
import numpy as np

ROOT = Path(__file__).resolve().parent
ARCHIVE = ROOT / 'TAC_UNIFIED_003AB.zip'
PIN = '34866019d1e0dd9293f82dcea61ce904cc88ce9168bf377fd79443d8d46cf73a'
N, K, C, B, T, SESSIONS = 256, 6, 32, 32, 1024, 128


def load(seed):
    assert hashlib.sha256(ARCHIVE.read_bytes()).hexdigest() == PIN
    with zipfile.ZipFile(ARCHIVE) as z, tempfile.TemporaryDirectory() as d:
        p = Path(d)/'prior.py'
        p.write_bytes(z.read('prior002/run.py'))
        spec = importlib.util.spec_from_file_location('prior_tac002', p)
        prior = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(prior)
        rows = json.loads(z.read('prior002/results.json'))['runs']
    row = next(r for r in rows if r['seed'] == seed)
    r = np.random.default_rng(seed)
    r.choice(len(list(itertools.permutations(range(8),3))),K,replace=False)
    while True:
        code = r.integers(0,2,(K,C),dtype=np.uint8)
        if min(np.count_nonzero(code[i]!=code[j]) for i,j in itertools.combinations(range(K),2)) >= 12:
            break
    train = r.permutation(N)[:96]
    ids = np.repeat(np.arange(K),len(train))
    cue = prior.noise(r,code[ids],2)
    sh = r.permutation(len(ids)); cue = cue[sh]
    cs,z = prior.cluster(cue)
    mapping = np.asarray(row['mapping']); assert np.all(mapping[z] == ids[sh])
    inv = np.argsort(mapping)
    learned = np.asarray(row['tables']['quad']); truth = np.asarray(row['true_table'])
    assert np.array_equal(learned,truth) and np.array_equal(cs,code[mapping])
    return prior,code[mapping],learned,truth


def flip(r, x, count):
    out = x.copy()
    idx = np.argpartition(r.random(x.shape,dtype=np.float32),13,axis=1)[:,:14]
    rr, cc = np.nonzero(np.arange(14)[None,:] < np.broadcast_to(count,(len(x),))[:,None])
    out[rr,idx[rr,cc]] ^= 1
    return out


def fixture(r, code, regime):
    # Time-major shape for causal group processing. Each row is one action decision.
    a = r.integers(K,size=(T,SESSIONS)); b = r.integers(16,size=a.shape)
    s = r.integers(N,size=(SESSIONS,16)); x = code[a.ravel()]
    views=[]
    if regime == 'correlated8':
        obs = flip(r,x,8); views = [obs]*5
    else:
        for _ in range(5):
            if regime == 'exact8': counts=8
            elif regime == 'variable789': counts=r.integers(7,10,size=len(x))
            elif regime == 'heavy': counts=np.where(r.random(len(x))<.20,14,8)
            elif regime == 'abrupt': counts=np.where((np.arange(len(x))>=len(x)//2)&(r.random(len(x))<.20),14,8)
            else: raise ValueError(regime)
            views.append(flip(r,x,counts))
    return a,b,s,views


def distances(x, cs):
    # count mismatches; no label/goal side channel
    return np.count_nonzero(x[:,None,:]!=cs[None,:,:],axis=2).astype(np.int16)


def fixed(views,cs):
    d=sum(distances(v,cs) for v in views)
    return d.argmin(1), np.full(len(d),5), []


def certificate(views,cs):
    n=len(views[0]); live=np.arange(n); ok=np.ones((n,K),bool)
    p=np.full(n,-1,int); reads=np.zeros(n,int)
    for i,v in enumerate(views):
        if not len(live): break
        ok[live] &= distances(v[live],cs)==8
        reads[live]+=1
        q=ok[live].sum(1); single=q==1
        p[live[single]]=ok[live[single]].argmax(1)
        done=(q<=1)|(i==4); live=live[~done]
    return p,reads,[]


def quantile(a):
    ar=np.sort(np.array(a)); return int(ar[min(len(ar)-1,math.ceil(.95*(len(ar)+1))-1)])


def adaptive(views,cs):
    n=len(views[0]); p=np.full(n,-1,int); reads=np.zeros(n,int); history=collections.deque(maxlen=512)
    q=8; ledger=[]; gsize=64*SESSIONS
    for start in range(0,n,gsize):
        end=min(start+gsize,n); size=end-start
        d=distances(views[0][start:end],cs); first=d.copy(); acc=d.copy()
        reads[start:end]=1; live=np.arange(size)
        audit=(np.arange(start,end)%16)==0
        force=audit|(d.min(1)>q)
        for i in range(5):
            if not len(live): break
            if i:
                acc[live]+=distances(views[i][start:end][live],cs);reads[start+live]+=1
            z=acc[live]; sort=np.partition(z,1,axis=1)[:,:2]
            done=((sort[:,1]-sort[:,0]>=max(10,2*q-6))&~force[live])|(i==4)
            p[start+live[done]]=z[done].argmin(1)
            live=live[~done]
        # all audit outcomes completed; only affects subsequent groups
        residual=first[np.flatnonzero(audit),p[start:end][audit]]
        history.extend(map(int,residual)); q_next=quantile(history)
        ledger.append({'start':int(start),'end':int(end),'q':q,'q_next':q_next,'audit_count':int(audit.sum()),'audit_first_disagreement':float(np.mean(first[audit].argmin(1)!=p[start:end][audit])),'reads':float(reads[start:end].mean())})
        q=q_next
    assert np.all(p>=0)
    return p,reads,ledger


def execute_chain(tab,s,b,a):
    s=s.copy(); rr=np.arange(len(s))
    for j in range(len(a)):
        s[rr,b[j]]=tab[a[j],s[rr,b[j]]]
    return s


def run_a(code,tab,truth,seed):
    arms={}
    for ri,regime in enumerate(('exact8','variable789','heavy','correlated8','abrupt')):
        a,b,s,views=fixture(np.random.default_rng(seed+50000+ri*1000),code,regime)
        gold=execute_chain(truth,s,b,a); row={}
        for name,method in [('certificate',certificate),('fixed5',fixed),('adaptive',adaptive)]:
            t=time.perf_counter_ns();p,reads,ledger=method(views,code);p=p.reshape(a.shape)
            predicted=execute_chain(tab,s,b,np.maximum(p,0));elapsed=(time.perf_counter_ns()-t)/1e6
            err=p!=a; abstain=p<0
            for l in ledger:
                l['action_errors']=int(err.ravel()[l['start']:l['end']].sum())
            row[name]={'actions':int(a.size),'action_errors':int(err.sum()),'wrong':int((err&~abstain).sum()),'abstentions':int(abstain.sum()),'chains':SESSIONS,'exact_actions':int(np.all(~err,axis=0).sum()),'final_correct':int((np.all(predicted==gold,axis=1)&~abstain.any(0)).sum()),'reads':float(reads.mean()),'elapsed_ms':elapsed,'groups':ledger}
        arms[regime]=row
    return arms


class World:
    """Constructed fixture with inaccessible-to-planner edges accessed only by interventions."""
    def __init__(self,table,edges):
        self.table=table;self.out={i:j for i,j in edges};self.ins={j:i for i,j in edges};self.queries=0
    def step(self,state,b,a):
        x=np.array(state,copy=True);self.queries+=int(np.prod(x.shape[:-1]))
        old=x[...,b].copy(); action=a
        if a==1 and b in self.ins:
            action=np.where(x[...,self.ins[b]]&1,2,1)
        x[...,b]=self.table[action,old]
        if a==0 and b in self.out:x[...,self.out[b]]^=(old&1)
        return x


def discover(world,tab):
    writes={};reads={};alt={}
    # Known action/bit family, exact deterministic interventions.
    for i in range(B):
        x=np.zeros((2,B),int);x[1,i]=1;y=world.step(x,i,0)
        diff=y^x; diff[:,i]=0
        for j in np.flatnonzero(diff[1]):
            if diff[0,j]!=0 or diff[1,j]!=1: raise ValueError('unexpected cross-write mask')
            if i in writes:raise ValueError('multiple writes')
            writes[i]=int(j)
    for j in range(B):
        zero=np.zeros((N,B),int);zero[:,j]=np.arange(N)
        baseline=world.step(zero,j,1)[:,j]
        if not np.array_equal(baseline,tab[1]):raise ValueError('unexpected baseline')
        for i in range(B):
            if i==j:continue
            x=zero.copy();x[:,i]=1;y=world.step(x,j,1)[:,j]
            if not np.array_equal(y,baseline):
                if j in reads:raise ValueError('multiple reads')
                candidates=[a for a in range(K) if np.array_equal(tab[a],y)]
                if len(candidates)!=1:raise ValueError('nonidentifiable alternate table')
                reads[j]=i;alt[j]=candidates[0]
    return writes,reads,alt


def matching(writes,reads):
    # supported graph: disjoint directed pairs with exactly the agreed read/write edge
    if len(writes)!=B//2 or len(reads)!=B//2 or any(reads.get(j)!=i for i,j in writes.items()):return None
    if len(set(writes)|set(writes.values()))!=B or set(writes)&set(writes.values()):return None
    return sorted(writes.items())


def pair_bfs(tab,x,y,gx,gy,alt):
    src=x*256+y;goal=gx*256+gy
    q=collections.deque([src]);prev={src:None};exp=0
    while q:
        cur=q.popleft();exp+=1
        if cur==goal:
            plan=[]
            while cur!=src:cur,b,a=prev[cur];plan.append((b,a))
            return plan[::-1],exp
        i,j=divmod(cur,256)
        for a in range(K):
            ni=int(tab[a,i]);nj=j^((i&1) if a==0 else 0);nxt=ni*256+nj
            if nxt not in prev:prev[nxt]=(cur,0,a);q.append(nxt)
            nj=int(tab[alt if a==1 and i&1 else a,j]);nxt=i*256+nj
            if nxt not in prev:prev[nxt]=(cur,1,a);q.append(nxt)
    return None,exp


def run_b(prior,tab,truth,seed):
    r=np.random.default_rng(seed+60000);order=r.permutation(B);edges=list(zip(map(int,order[::2]),map(int,order[1::2])))
    world=World(truth,edges);t=time.perf_counter_ns()
    writes,reads,alt=discover(world,tab);discovery_ms=(time.perf_counter_ns()-t)/1e6;discovery_queries=world.queries
    pairs=matching(writes,reads);expected=set(edges);learned=set(writes.items())|{(i,j) for j,i in reads.items()}
    # The original random operators can induce duplicate tables. Do not silently accept an unidentifiable alternative.
    if pairs is None:raise ValueError('unsupported matching')
    # Actual observed cyclic controls: use a directed 2-cycle and compare matching guard.
    cyclic=World(truth,[(0,1),(1,0)]+[(i,i+1) for i in range(2,B-1,2)])
    cw,cr,ca=discover(cyclic,tab);cyclic_rejected=matching(cw,cr) is None
    count=64; stats={name:{'correct':0,'planned':0,'lengths':[],'elapsed_ms':[],'expansions':[]} for name in ('local','topological','joint')}
    for _ in range(count):
        start=r.integers(N,size=B);goal=start.copy()
        for step in range(8):
            for block in r.permutation(B):goal=world.step(goal,int(block),int(r.integers(K)))
        for name in stats:
            t=time.perf_counter_ns(); plan=[]; expanded=0; valid=True
            if name=='local':
                for i in range(B):
                    p,e=prior.bfs(tab,int(start[i]),int(goal[i]));expanded+=e
                    if p is None:valid=False;break
                    plan.extend((i,a) for a in p)
            else:
                for i,j in pairs:
                    if name=='joint':
                        p,e=pair_bfs(tab,int(start[i]),int(start[j]),int(goal[i]),int(goal[j]),alt[j]);expanded+=e
                        if p is None:valid=False;break
                        plan.extend((i if b==0 else j,a) for b,a in p)
                    else:
                        p,e=prior.bfs(tab,int(start[i]),int(goal[i]));expanded+=e
                        if p is None:valid=False;break
                        si=int(start[i]);sj=int(start[j]);plan.extend((i,a) for a in p)
                        for a in p:
                            if a==0:sj^=si&1
                            si=int(tab[a,si])
                        local=tab.copy()
                        if si&1:local[1]=tab[alt[j]]
                        p,e=prior.bfs(local,sj,int(goal[j]));expanded+=e
                        if p is None:valid=False;break
                        plan.extend((j,a) for a in p)
            # Actual execution; verification compares only after constructing plan.
            result=start.copy()
            if valid:
                for b,a in plan:result=world.step(result,b,a)
            rec=stats[name];rec['planned']+=int(valid);rec['correct']+=int(valid and np.array_equal(result,goal));rec['lengths'].append(len(plan) if valid else -1);rec['elapsed_ms'].append((time.perf_counter_ns()-t)/1e6);rec['expansions'].append(expanded)
    return {'goals':count,'discovery_ms':discovery_ms,'discovery_queries':discovery_queries,'edge_tp':len(learned&expected),'edge_fp':len(learned-expected),'edge_fn':len(expected-learned),'expected_edges':sorted(expected),'learned_edges':sorted(learned),'alternate':alt,'cyclic_rejected':cyclic_rejected,'clusters':len(pairs),'arms':stats}


def main():
    seed=int(os.environ['TRAINLAB_SEED']);out=Path(os.environ['TRAINLAB_OUTPUT_DIR']);out.mkdir(exist_ok=True,parents=True)
    prior,code,tab,truth=load(seed)
    a=run_a(code,tab,truth,seed);b=run_b(prior,tab,truth,seed)
    gates={r:(a[r]['adaptive']['final_correct']/SESSIONS>=.99 and a[r]['adaptive']['reads']<3 and (a[r]['fixed5']['final_correct']-a[r]['adaptive']['final_correct'])/SESSIONS<=.01) for r in ('exact8','variable789')}
    gates['dependency']=b['edge_fp']==0 and b['edge_fn']==0 and b['arms']['joint']['correct']==b['goals'] and b['cyclic_rejected']
    result={'seed':seed,'archive_sha256':PIN,'numpy':np.__version__,'scope':'CONSTRUCTED_NO_FIT_NOT_QUALIFIED','a':a,'b':b,'gates':gates}
    (out/'results.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    metrics={'adaptive_exact8_accuracy':a['exact8']['adaptive']['final_correct']/SESSIONS,'adaptive_variable_accuracy':a['variable789']['adaptive']['final_correct']/SESSIONS,'adaptive_exact8_reads':a['exact8']['adaptive']['reads'],'adaptive_variable_reads':a['variable789']['adaptive']['reads'],'joint_goal_accuracy':b['arms']['joint']['correct']/b['goals'],'edge_errors':b['edge_fp']+b['edge_fn'],'passed_gates':sum(gates.values())}
    (out/'metrics.json').write_text(json.dumps(metrics,indent=2,sort_keys=True)+'\n');print(seed,metrics,gates,flush=True)

if __name__=='__main__':main()
