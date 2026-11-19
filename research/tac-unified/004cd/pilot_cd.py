"""Constructed no-fit TAC-UNIFIED-004C/D diagnostic, executed by TrainLab."""
import collections
import json
import os
from pathlib import Path
import time
import numpy as np
import prior004 as old

B, N, K, T, S = old.B, old.N, old.K, old.T, old.SESSIONS


def targeted(world, tab):
    witnesses = np.flatnonzero(tab[1] != tab[2])
    if not len(witnesses):
        raise ValueError('no distinguishing local witness')
    w = int(witnesses[0]); writes = {}; reads = {}; alt = {}
    for i in range(B):
        x = np.zeros((2, B), int); x[1, i] = 1
        dy = world.step(x, i, 0) ^ x; dy[:, i] = 0
        for j in np.flatnonzero(dy[1]):
            if dy[0, j] != 0 or dy[1, j] != 1 or i in writes:
                raise ValueError('unsupported write')
            writes[i] = int(j)
    screens = []
    for j in range(B):
        parents = [i for i in range(B) if i != j]
        x = np.zeros((B, B), int); x[:, j] = w
        x[np.arange(1, B), parents] = 1
        got = world.step(x, j, 1)[:, j]
        if got[0] != tab[1, w]: raise ValueError('bad witness baseline')
        candidates = [i for i, y in zip(parents, got[1:]) if y != got[0]]
        screens.append({'target': j, 'candidates': candidates})
        if len(candidates) > 1: raise ValueError('multiple visible parents')
        for i in candidates:
            zero = np.zeros((N, B), int); zero[:, j] = np.arange(N)
            base = world.step(zero, j, 1)[:, j]
            if not np.array_equal(base, tab[1]): raise ValueError('bad sweep baseline')
            x = zero.copy(); x[:, i] = 1
            y = world.step(x, j, 1)[:, j]
            alts = [a for a in range(K) if np.array_equal(tab[a], y)]
            if len(alts) != 1 or alts[0] == 1: raise ValueError('unsupported alternate')
            reads[j] = i; alt[j] = alts[0]
    return writes, reads, alt, w, screens


class DormantWorld(old.World):
    def __init__(self, tab, edges, hidden):
        super().__init__(tab, edges); self.hidden = hidden

    def step(self, state, b, a):
        i, j, val = self.hidden
        x = super().step(state, b, a)
        if b == j and a == 1:
            s = np.asarray(state)
            on = ((s[..., i] & 1) != 0) & (s[..., j] == val)
            x[..., b] = np.where(on, self.table[2, s[..., j]], x[..., b])
        return x


def probe(method, world, tab):
    start = time.perf_counter_ns(); result = None; err = None
    try:
        result = method(world, tab)
        pairs = old.matching(*result[:2]); accepted = pairs is not None
    except ValueError as exc:
        accepted = False; err = str(exc)
    return result, {'accepted': accepted, 'error': err, 'queries': world.queries,
                    'elapsed_ms': (time.perf_counter_ns() - start)/1e6,
                    'writes': sorted(result[0].items()) if result else [],
                    'reads': sorted(result[1].items()) if result else [],
                    'alternates': result[2] if result else {}}


def run_c(tab, truth, seed):
    r = np.random.default_rng(seed + 70000)
    order = r.permutation(B); edges = list(zip(map(int, order[::2]), map(int, order[1::2])))
    full, f = probe(old.discover, old.World(truth, edges), tab)
    short, t = probe(targeted, old.World(truth, edges), tab)
    exact = bool(full and short and full == short[:3] and set(short[0].items()) == set(edges)
                 and set((i,j) for j,i in short[1].items()) == set(edges))
    cases = {}; ro = old.World(truth, edges); del ro.out[edges[0][0]]
    cases['read_only'] = probe(targeted, ro, tab)[1]
    cycle = old.World(truth, [(0,1),(1,0)] + [(i,i+1) for i in range(2,B-1,2)])
    cases['cycle'] = probe(targeted, cycle, tab)[1]
    # A hidden second read is deliberately absent at the chosen witness.
    ds = np.flatnonzero(tab[1] != tab[2]); i0,j0 = edges[0]
    h = next(i for i in range(B) if i not in (i0,j0)); hidden = (h,j0,int(ds[-1]))
    if len(ds) < 2: raise ValueError('seed lacks dormant falsifier')
    cases['dormant_targeted'] = probe(targeted, DormantWorld(truth,edges,hidden),tab)[1]
    cases['dormant_full'] = probe(old.discover, DormantWorld(truth,edges,hidden),tab)[1]
    # Verify the concrete missed residual with the hidden parent active and original parent inactive.
    x = np.zeros(B,int); x[j0]=hidden[2]; x[h]=1
    actual = DormantWorld(truth, edges, hidden).step(x,j0,1)
    expected = old.World(tab, edges).step(x,j0,1)
    missed = bool(not np.array_equal(actual,expected) and cases['dormant_targeted']['accepted'])
    world = old.World(truth,edges); goals=[]
    for _ in range(64):
        initial=r.integers(N,size=B); goal=initial.copy()
        for __ in range(8):
            for block in r.permutation(B): goal=world.step(goal,int(block),int(r.integers(K)))
        start=time.perf_counter_ns(); plan=[]; expanded=0; valid=bool(short and t['accepted'])
        if valid:
            for i,j in old.matching(*short[:2]):
                p,e=old.pair_bfs(tab,int(initial[i]),int(initial[j]),int(goal[i]),int(goal[j]),short[2][j]); expanded+=e
                if p is None: valid=False; break
                plan.extend((i if b==0 else j,a) for b,a in p)
        result=initial.copy()
        if valid:
            for b,a in plan: result=world.step(result,b,a)
        goals.append({'correct':bool(valid and np.array_equal(result,goal)), 'planned':valid,
                      'length':len(plan) if valid else -1, 'expansions':expanded,
                      'elapsed_ms':(time.perf_counter_ns()-start)/1e6})
    n=sum(g['correct'] for g in goals)
    gate=exact and t['queries']<=.1*f['queries'] and n==64 and not cases['read_only']['accepted'] and not cases['cycle']['accepted']
    return {'full':f,'targeted':t,'exact':exact,'goals':goals,'correct':n,'cases':cases,
            'dormant_hidden':hidden,'dormant_false_accept':missed,'gate':bool(gate),
            'witness':short[3] if short else None,'screens':short[4] if short else []}


def forced_secondary(views,cs):
    # Match 004A's chronological audit; forbid a one-view decision.
    n=len(views[0]); p=np.full(n,-1,int); reads=np.zeros(n,int)
    hist=collections.deque(maxlen=512); q=8; ledger=[]; gsize=64*S
    for start in range(0,n,gsize):
        end=min(start+gsize,n); live=np.arange(end-start)
        first=old.distances(views[0][start:end],cs); acc=first.copy()
        reads[start:end]=1
        audit=(np.arange(start,end)%16)==0; force=audit|(first.min(1)>q)
        for i in range(5):
            if not len(live): break
            if i:
                acc[live]+=old.distances(views[i][start:end][live],cs); reads[start+live]+=1
            z=acc[live]; sort=np.partition(z,1,axis=1)[:,:2]
            done=(((sort[:,1]-sort[:,0]>=max(10,2*q-6))&~force[live])&(i>=1))|(i==4)
            p[start+live[done]]=z[done].argmin(1); live=live[~done]
        hist.extend(map(int,first[np.flatnonzero(audit),p[start:end][audit]]))
        nxt=old.quantile(hist)
        ledger.append({'start':start,'end':end,'q':q,'q_next':nxt,'audits':int(audit.sum()),'reads':float(reads[start:end].mean())})
        q=nxt
    assert np.all(p>=0) and np.all(reads>=2)
    return p,reads,ledger


def run_d(code,tab,truth,seed):
    arms={}
    for ri,regime in enumerate(('exact8','variable789','heavy','shared_all')):
        r=np.random.default_rng(seed+80000+ri*1000)
        a=r.integers(K,size=(T,S)); b=r.integers(16,size=a.shape); s=r.integers(N,size=(S,16))
        x=code[a.ravel()]; primary=old.flip(r,x,8); views=[primary]
        for _ in range(4):
            if regime=='shared_all': v=primary
            else:
                count=8 if regime=='exact8' else r.integers(7,10,len(x)) if regime=='variable789' else np.where(r.random(len(x))<.2,14,8)
                v=old.flip(r,x,count)
            views.append(v)
        gold=old.execute_chain(truth,s,b,a); row={}
        for name,method,offered in [('repeat_primary',old.fixed,[primary]*5),('fixed_diverse',old.fixed,views),('adaptive_diverse',old.adaptive,views),('forced_secondary',forced_secondary,views)]:
            start=time.perf_counter_ns(); p,reads,ledger=method(offered,code); p=p.reshape(a.shape)
            predicted=old.execute_chain(tab,s,b,np.maximum(p,0)); elapsed=(time.perf_counter_ns()-start)/1e6
            err=p!=a; abstain=p<0
            for l in ledger: l['action_errors']=int(err.ravel()[l['start']:l['end']].sum())
            row[name]={'actions':int(a.size),'action_errors':int(err.sum()),'abstentions':int(abstain.sum()),'chains':S,'exact_actions':int(np.all(~err,axis=0).sum()),'final_correct':int((np.all(predicted==gold,axis=1)&~abstain.any(0)).sum()),'reads':float(reads.mean()),'secondary_fraction':float(np.mean(reads>=2)) if name!='repeat_primary' else 0.,'elapsed_ms':elapsed,'groups':ledger}
        arms[regime]=row
    return arms


def main():
    seed=int(os.environ['TRAINLAB_SEED']); out=Path(os.environ['TRAINLAB_OUTPUT_DIR']);out.mkdir(parents=True,exist_ok=True)
    _,code,tab,truth=old.load(seed); c=run_c(tab,truth,seed); d=run_d(code,tab,truth,seed)
    gates={'dependency':c['gate']}
    for regime in ('exact8','variable789'):
        f=d[regime]['forced_secondary']; baseline=d[regime]['fixed_diverse']
        gates[regime]=f['final_correct']/S>=.99 and (baseline['final_correct']-f['final_correct'])/S<=.01 and f['reads']<4.5
    result={'seed':seed,'scope':'CONSTRUCTED_NO_FIT_NOT_QUALIFIED','archive_sha256':old.PIN,'c':c,'d':d,'gates':gates}
    metrics={'passed_gates':sum(gates.values()),'targeted_queries':c['targeted']['queries'],'full_queries':c['full']['queries'],'joint_accuracy':c['correct']/64,'diverse_exact8_accuracy':d['exact8']['forced_secondary']['final_correct']/S,'diverse_variable_accuracy':d['variable789']['forced_secondary']['final_correct']/S,'diverse_exact8_reads':d['exact8']['forced_secondary']['reads'],'diverse_variable_reads':d['variable789']['forced_secondary']['reads'],'dormant_false_accept':int(c['dormant_false_accept'])}
    (out/'results.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    (out/'metrics.json').write_text(json.dumps(metrics,indent=2,sort_keys=True)+'\n'); print(seed,metrics,gates,flush=True)

if __name__=='__main__': main()
