"""Registered no-fit real-input equivalence and paired execution timings."""
import hashlib,json,platform,random,subprocess,time
from pathlib import Path
import numpy as np
import torch
from scripts.me080_train_world import integrity,frozen,load_original,evaluation_origins,batch,SEEDS,OUT,sha
from tac_osm.variants.streaming_mixture_world import StreamingMixtureWorld,distribution_loss,quantile
from tac_osm.variants.reusable_execution import BatchedMixtureWorld,mixture_quantiles

ROOT=Path('research/computation-reuse')


def main():
    torch.set_num_threads(1);integrity();frozen()
    reg=json.loads((ROOT/'REGISTRATION.json').read_text())
    for name in ('src/tac_osm/variants/reusable_execution.py','scripts/benchmark_reusable_execution.py','research/computation-reuse/REGISTRATION.json'):
        original=subprocess.check_output(['git','show','HEAD:'+name])
        if hashlib.sha256(original).hexdigest()!=sha(name):raise RuntimeError('Freeze implementation/registration BEFORE timing')
    _,fit,report,identified,evidence=load_original(Path('/tmp/me077-original-source'))
    queries,_=evaluation_origins(report,identified);normalizer=float(np.mean(identified['delta_scale']*identified['state_scale']))
    results=[];rng=random.Random(81081)
    for seed in SEEDS:
        folder=OUT/f'seed-{seed}';weights=torch.load(folder/'final.pt',map_location='cpu',weights_only=True)['weights']
        reference=StreamingMixtureWorld(identified).eval();optimized=BatchedMixtureWorld(identified).eval()
        reference.load_state_dict(weights,strict=True);optimized.load_state_dict(weights,strict=True)
        assert sum(p.numel() for p in optimized.parameters())==99816
        maxima={};count=0
        with torch.no_grad():
            for i in range(0,len(queries),24):
                x,a,_,_=batch(report,queries[i:i+24]);old=reference(x,a);new=optimized(x,a);last=optimized.forecast_last(x,a)
                for key in old:
                    for label,value,expected in [('all',new[key],old[key]),('last',last[key],old[key][:,-1])]:
                        torch.testing.assert_close(value,expected,rtol=1e-5,atol=1e-6)
                        name=label+'/'+key;maxima[name]=max(maxima.get(name,0.),float((value-expected).abs().max()))
                expected=torch.stack([quantile({k:v[:,-1] for k,v in old.items()},p) for p in (.05,.5,.95)])
                joint=mixture_quantiles({k:v[:,-1] for k,v in old.items()})
                torch.testing.assert_close(joint,expected,rtol=1e-5,atol=1e-6)
                maxima['joint_quantile_inversion']=max(maxima.get('joint_quantile_inversion',0.),float((joint-expected).abs().max()))
                count+=len(x)
        # Real FIT batch, no optimizer construction or parameter update.
        task,ep=sorted(fit)[0];human=fit[(task,ep)]['human'];length=len(human)
        t=next(t for t in range(8,length-17) if not bool(human[t-8:t+17].any()))
        x,a,y,valid=batch(fit,[(task,ep,t)]*24)
        with torch.no_grad():raw={k:v[:,-1] for k,v in reference(x,a).items()}
        def forward(m):
            with torch.no_grad():return m(x,a)
        def backward(m):
            m.zero_grad(set_to_none=True)
            loss=distribution_loss(m(x,a),y,x,valid,torch.ones(24),m.state_scale,normalizer)
            loss.backward()
        def last_old():
            with torch.no_grad():return {k:v[:,-1] for k,v in reference(x,a).items()}
        def last_new():
            with torch.no_grad():return optimized.forecast_last(x,a)
        def quant_old():
            with torch.no_grad():return torch.stack([quantile(raw,p) for p in (.05,.5,.95)])
        def quant_new():
            with torch.no_grad():return mixture_quantiles(raw)
        workflows={'forward_all':(lambda:forward(reference),lambda:forward(optimized)),
                   'forward_backward_all':(lambda:backward(reference),lambda:backward(optimized)),
                   'last_prefix':(last_old,last_new),'three_quantiles':(quant_old,quant_new)}
        timings={}
        for name,functions in workflows.items():
            for _ in range(5):
                for fn in functions:fn()
            pairs=[]
            for _ in range(31):
                elapsed=[0.,0.];order=[0,1];rng.shuffle(order)
                for j in order:
                    begin=time.perf_counter_ns();functions[j]();elapsed[j]=(time.perf_counter_ns()-begin)/1e6
                pairs.append(elapsed)
            values=np.asarray(pairs);ratios=values[:,0]/values[:,1]
            draws=np.random.default_rng(seed).integers(31,size=(2000,31))
            timings[name]={'reference_median_ms':float(np.median(values[:,0])),
                'optimized_median_ms':float(np.median(values[:,1])),
                'median_paired_speedup':float(np.median(ratios)),
                'bootstrap95CI_paired_speedup':np.quantile(np.median(ratios[draws],1),[.025,.975]).tolist(),
                'reference_IQR_ms':np.quantile(values[:,0],[.25,.75]).tolist(),
                'optimized_IQR_ms':np.quantile(values[:,1],[.25,.75]).tolist(),'raw_paired_ms':pairs}
        for m in (reference,optimized):
            for key,value in m.state_dict().items():
                if not torch.equal(value,weights[key]):raise RuntimeError('Benchmark changed original model values')
        results.append({'seed':seed,'checkpoint_sha256':sha(folder/'final.pt'),'parameters':99816,
                        'actual_exposed_query_windows_checked':count,'agreement':'PASS','maximum_absolute_differences':maxima,'timing':timings})
        print('ORIGINAL_CHECKPOINT_BENCHMARK_COMPLETE',seed,flush=True)
    artifact={'study_id':'ENG_REUSE_001','operation':'NO_FIT','value_contract':'PASS on all original real-input windows; gradient contract verified separately in software tests',
       'original_source_sha256':evidence['original_160_episode_source_sha256'],
       'scope':'same5frozen models,40exposed episodes,960query windows per seed; benchmark FITbatch24/history8/threads1; no new prediction accuracy, long-memory or intelligence claim',
       'environment':{'python':platform.python_version(),'torch':torch.__version__,'platform':platform.platform(),'processor':platform.processor(),'threads':torch.get_num_threads()},
       'implementation_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
       'implementation_sha256':{p:sha(p) for p in ('src/tac_osm/variants/reusable_execution.py','scripts/benchmark_reusable_execution.py')},
       'registration_sha256':sha(ROOT/'REGISTRATION.json'),'results':results}
    path=ROOT/'ORIGINAL_EXECUTION_RESULTS.json'
    if path.exists():raise RuntimeError('Never overwrite original timings')
    path.write_text(json.dumps(artifact,sort_keys=True,indent=2)+'\n')
    for name in workflows:
        print(name,[(r['seed'],round(r['timing'][name]['median_paired_speedup'],3)) for r in results])

if __name__=='__main__':main()
