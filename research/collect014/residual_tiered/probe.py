"""Best-case exact-address storage test; no trainable model or query leakage."""
from pathlib import Path
import hashlib,json,time
import numpy as np

ROOT=Path(__file__).resolve().parent

def address(rng,n,d):
    theta=rng.uniform(-np.pi,np.pi,(n,d//2))
    return np.stack((np.cos(theta),np.sin(theta)),-1).reshape(n,d)/np.sqrt(d//2)

def stream(seed,n,regime):
    rng=np.random.default_rng(seed);background=1 if regime=='repeated_distractor' else n*3
    keys=address(rng,n+background,16);labels=rng.integers(0,8,n+background);records=[]
    for i in range(n):
        records.append((keys[i],np.eye(8)[labels[i]],i))
        if regime!='uniform_facts':
            for j in range(3):
                k=n if regime=='repeated_distractor' else n+3*i+j
                records.append((keys[k],np.eye(8)[labels[k]],k))
    return records,keys[:n],labels[:n]

def match(entries,key):
    if not entries:return None
    similarity=np.array([k@key for k,v in entries]);i=int(similarity.argmax())
    return i if similarity[i]>1-1e-10 else None

def simulate(records,queries,labels,capacity,policy,threshold,random_mask=None):
    states=np.zeros((2,8,16));cache=[];local=[];admissions=[];evictions=0;residuals=[]
    for index,(key,value,tag) in enumerate(records):
        states*=np.array([.98,.999])[:,None,None]
        read=(states@key).mean(0);error=float(np.linalg.norm(value-read));residuals.append(error)
        admit=(error>threshold) if policy=='residual' else policy=='admit_all' or (policy=='random_count_matched' and index in random_mask)
        # Any known overwrite invalidates a cached old value, even if not admitted.
        old=match(cache,key)
        if old is not None:cache.pop(old)
        if admit:
            admissions.append(index);cache.append((key.copy(),value.copy()))
            if len(cache)>capacity:cache.pop(0);evictions+=1
        residual=value-(states@key)
        states+=residual[:,:,None]*key[None,None,:]
        local.append((key.copy(),value.copy()))
        if len(local)>8:local.pop(0)
    predictions=[];exact=[];routes=[]
    for key,label in zip(queries,labels):
        pos=match(local,key)
        if pos is not None:value=local[pos][1];route='local'
        else:
            pos=match(cache,key)
            if pos is not None:value=cache[pos][1];route='cache'
            else:value=(states@key).mean(0);route='compressed'
        predictions.append(int(value.argmax()));exact.append(bool(np.max(np.abs(value-np.eye(8)[label]))<1e-9));routes.append(route)
    correct=(np.array(predictions)==labels)
    cached=sum(match(cache,key) is not None for key in queries)
    return dict(class_accuracy=float(correct.mean()),exact_value_recall=float(np.mean(exact)),complete_class_map=bool(correct.all()),
        admissions=admissions,admission_count=len(admissions),eviction_count=evictions,distinct_cached_facts=cached,
        predictions=predictions,targets=labels.tolist(),routes=routes,prewrite_residuals=residuals,
        numeric_state_bytes=states.nbytes+sum(k.nbytes+v.nbytes for k,v in cache+local))

def counterexample():
    e1=np.array([1.,0.]);e2=np.array([0.,1.]);near=np.array([.95,np.sqrt(1-.95**2)])
    s=np.zeros((2,2));cache=[];errors=[];admitted=[]
    records=[(e1,np.array([1.,0.])),(e2,np.array([0.,1.])),(e1,np.array([1.,0.])),(near,np.array([0.,1.]))]
    for k,v in records:
        error=float(np.linalg.norm(v-s@k));errors.append(error);admitted.append(error>.75)
        old=match(cache,k)
        if old is not None:cache.pop(old)
        if error>.75:
            cache.append((k,v))
            if len(cache)>1:cache.pop(0)
        s+=np.outer(v-s@k,k)
    read=s@e1
    assert errors[2]==0 and not admitted[2] and read.argmax()!=0 and match(cache,e1) is None
    return dict(prewrite_errors=errors,admitted=admitted,final_query_read=read.tolist(),target=[1.,0.],
        missing_from_exact_cache=True,final_class_wrong=True,
        interpretation='Zero current write residual does not guarantee future survival or future irrelevance. This is an engineered counterexample, not a frequency estimate.')

def main():
    protocol=json.loads((ROOT/'PROTOCOL.json').read_text());rows=[];tic=time.perf_counter()
    for seed in protocol['seeds']:
        for n in protocol['active_fact_keys']:
            for regime in protocol['regimes']:
                for history in range(protocol['histories_per_seed_cell']):
                    code=seed*100000+n*100+history
                    records,queries,labels=stream(code,n,regime)
                    for cap in protocol['exact_capacities']:
                        residual=simulate(records,queries,labels,cap,'residual',protocol['threshold'])
                        rng=np.random.default_rng(code+cap*100000000)
                        mask=set(rng.choice(len(records),size=residual['admission_count'],replace=False).tolist())
                        for policy in protocol['policies']:
                            result=residual if policy=='residual' else simulate(records,queries,labels,cap,policy,protocol['threshold'],mask)
                            if policy=='random_count_matched':assert result['admission_count']==residual['admission_count']
                            rows.append(dict(seed=seed,N=n,regime=regime,history=history,capacity=cap,policy=policy,**result))
    summaries=[]
    for n in protocol['active_fact_keys']:
        for regime in protocol['regimes']:
            for cap in protocol['exact_capacities']:
                for policy in protocol['policies']:
                    group=[r for r in rows if (r['N'],r['regime'],r['capacity'],r['policy'])==(n,regime,cap,policy)]
                    summaries.append(dict(N=n,regime=regime,capacity=cap,policy=policy,histories=len(group),
                        **{k:float(np.mean([r[k] for r in group])) for k in ('class_accuracy','exact_value_recall','complete_class_map','admission_count','eviction_count','distinct_cached_facts','numeric_state_bytes')}))
    result=dict(status='GENERATED_COMPONENT_FALSIFICATION_NOT_TRAINED_LM',protocol=protocol,
        protocol_sha256=hashlib.sha256((ROOT/'PROTOCOL.json').read_bytes()).hexdigest(),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),numpy=np.__version__,optimizer_updates=0,
        counterexample=counterexample(),summaries=summaries,rows=rows,elapsed_seconds=time.perf_counter()-tic)
    (ROOT/'RESULTS.json').write_text(json.dumps(result,indent=2));print(json.dumps({'status':result['status'],'rows':len(rows),'summaries':len(summaries),'seconds':result['elapsed_seconds'],'counterexample':result['counterexample']},indent=2))

if __name__=='__main__':main()
