"""Real-data native execution engineering; no new predictive capability fit."""
import argparse,json,math,platform,resource,time
from pathlib import Path
import numpy as np
import torch
import scripts.me038_research as old
from tac_osm.me039_native_execution import build_model
from scripts.me038_audit import PhysicalForecast

ROOT=old.ROOT;OUT=ROOT/'results/me039';REPORT=ROOT/'research/mathematical-engineering/me039'
BACKENDS=('eager','scripted');SNAPS=(1,16,64)
sha,dump=old.sha,old.dump


def source():
    for item in json.loads((REPORT/'dependencies.json').read_text())['files']:
        assert sha(ROOT/item['path'])==item['sha256'],item['path']
    frozen=json.loads((ROOT/'results/me038/frozen.json').read_text())
    rows=[r for r in frozen['fits'] if r['arm']=='innovation_tac' and r['status']=='PASS']
    assert len(rows)==60
    for r in rows:assert sha(ROOT/r['selected']['checkpoint'])==r['selected']['sha256']
    return rows


def parameters(model):return dict(sorted(model.named_parameters()))

def flatten(model,gradient=False):
    return torch.cat([(p.grad if gradient else p).detach().flatten() for p in parameters(model).values()]).clone().numpy()


def continuation(row,rep,backend):
    key=f"{row['dataset']}-h{row['horizon']}-s{row['seed']}-r{rep}-{backend}"
    folder=OUT/'training'/key;folder.mkdir(parents=True,exist_ok=True)
    z,meta=old.data(row['dataset']);h=row['horizon'];seed=row['seed']
    start=time.process_time();model=build_model(h,seed,backend)
    model.load_state_dict(torch.load(ROOT/row['selected']['checkpoint'],weights_only=False)['state_dict'])
    extra=[p for n,p in model.named_parameters() if not n.startswith('ar.')]
    opt=torch.optim.AdamW([{'params':list(model.ar.parameters()),'lr':.0001},{'params':extra,'lr':.001}],weight_decay=.0001)
    gen=torch.Generator().manual_seed(seed+3000+rep)
    construction=time.process_time()-start;trace=[];snapshots={};checks=0.
    for step in range(1,65):
        t=time.process_time();x,y=old.batch(z,torch.randint(96,meta['split_edges'][0]-h+1,(4,),generator=gen),h)
        model.train();opt.zero_grad(set_to_none=True);pred,delta,_=model.parts(x)
        mse=(pred-y).square().mean();loss=mse+10*model.anchor_loss()+.01*delta.square().mean()
        assert torch.isfinite(loss)
        loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.);assert torch.isfinite(norm)
        opt.step();elapsed=time.process_time()-t
        t=time.process_time();trace.append({'step':step,'loss':loss.item(),'MSE':mse.item(),'gradient_norm':norm.item(),'CPU_seconds':elapsed})
        if step in SNAPS:
            grad=flatten(model,True);weights=flatten(model)
            np.savez(folder/f'step{step}.npz',gradient=grad,parameters=weights,forecast_prefix=pred.detach().flatten()[:128].numpy())
            snapshots[step]=(grad,weights)
        checks+=time.process_time()-t
    torch.save({'state_dict':model.state_dict(),'optimizer':opt.state_dict(),'sampler_rng':gen.get_state(),
        'model_rng':torch.get_rng_state(),'step':64,'engineering_only_continuation':True},folder/'final.pt')
    p=parameters(model);offsets=[];i=0
    for name,tensor in p.items():offsets.append({'name':name,'start':i,'end':i+tensor.numel()});i+=tensor.numel()
    result={'key':key,'dataset':row['dataset'],'horizon':h,'seed':seed,'repeat':rep,'backend':backend,
        'original_checkpoint':row['selected'],'steps':64,'windows':256,'parameters':i,'parameter_offsets':offsets,
        'construction_compile_load_optimizer_CPU':construction,'cold16_CPU':construction+sum(r['CPU_seconds'] for r in trace[:16]),
        'steady48_CPU':sum(r['CPU_seconds'] for r in trace[16:]),'total64_CPU':construction+sum(r['CPU_seconds'] for r in trace),
        'check_CPU_outside_timing':checks,'trace':trace,'final_checkpoint':str((folder/'final.pt').relative_to(ROOT)),
        'final_sha256':sha(folder/'final.pt'),'snapshots':[{'step':s,'path':str((folder/f'step{s}.npz').relative_to(ROOT)),
            'sha256':sha(folder/f'step{s}.npz')} for s in SNAPS],
        'peak_process_RSS_KiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'status':'PASS'}
    dump(folder/'record.json',result);print('REPLAY',key,round(result['total64_CPU'],3),round(result['steady48_CPU'],3),flush=True)
    return result,snapshots


def train_benchmark():
    torch.set_num_threads(1);rows=source();OUT.mkdir(parents=True,exist_ok=True)
    # Warm architecture-independent optimizer import once, and record its cost.
    t=time.process_time();torch.optim.AdamW([torch.nn.Parameter(torch.zeros(1))]);startup=time.process_time()-t
    cases=[r for r in rows if r['lane']=='exposure' and r['dataset'] in ('ETTh1','electricity') and r['horizon'] in (96,720)]
    assert len(cases)==8;records=[];pairs=[];u=2**-24;tol=4*96*u/(1-96*u)
    for ci,r in enumerate(cases):
        for rep in range(3):
            order=BACKENDS if (ci+rep)%2==0 else BACKENDS[::-1];done={}
            for backend in order:
                record,snap=continuation(r,rep,backend);records.append(record);done[backend]=(record,snap)
            a,sa=done['eager'];b,sb=done['scripted'];deviations=[];perparameter=[]
            for step in SNAPS:
                ga,pa=sa[step];gb,pb=sb[step];error=ga.astype(np.float64)-gb
                rel=np.linalg.norm(error)/max(np.linalg.norm(ga.astype(np.float64)),1e-12)
                mask=np.concatenate([np.full(v['end']-v['start'],v['name'].startswith('memory.')) for v in a['parameter_offsets']])
                core=np.linalg.norm(error[mask])/max(np.linalg.norm(ga[mask].astype(np.float64)),1e-12)
                dev={'step':step,'gradient_relative_L2':float(rel),'memory_gradient_relative_L2':float(core),
                    'gradient_max_abs':float(np.max(np.abs(error))),'parameter_relative_L2':float(np.linalg.norm(pa.astype(np.float64)-pb)/max(np.linalg.norm(pa.astype(np.float64)),1e-12))}
                deviations.append(dev)
                for v in a['parameter_offsets']:
                    sl=slice(v['start'],v['end']);perparameter.append({'step':step,'name':v['name'],'gradient_max_abs':float(np.max(np.abs(error[sl])))})
            ok=all(d['gradient_relative_L2']<=tol and d['memory_gradient_relative_L2']<=tol and d['parameter_relative_L2']<=64*tol for d in deviations)
            pairs.append({'case':r['key'],'repeat':rep,'eager':a['key'],'scripted':b['key'],'order':list(order),
                'steady_speedup':a['steady48_CPU']/b['steady48_CPU'],'total_speedup':a['total64_CPU']/b['total64_CPU'],
                'deviations':deviations,'per_parameter':perparameter,'agreement_pass':ok})
    case_summary=[]
    for r in cases:
        pp=[p for p in pairs if p['case']==r['key']]
        case_summary.append({'case':r['key'],'steady_speedup':float(np.median([p['steady_speedup'] for p in pp])),
            'total_speedup':float(np.median([p['total_speedup'] for p in pp]))})
    steady=math.exp(np.mean([math.log(r['steady_speedup']) for r in case_summary]));total=math.exp(np.mean([math.log(r['total_speedup']) for r in case_summary]))
    receipt={'status':'PASS' if all(p['agreement_pass'] for p in pairs) else 'NUMERICAL_FAIL',
        'protocol_sha256':sha(REPORT/'PROTOCOL.md'),'model_sha256':sha(ROOT/'src/tac_osm/me039_native_execution.py'),
        'runner_sha256':sha(Path(__file__)),'cases':case_summary,'pairs':pairs,'records':records,
        'steady_geometric_speedup':steady,'total_geometric_speedup':total,
        'steady_target_pass':bool(steady>=1.2),'cold_inclusive_target_pass':bool(total>=1.1),
        'roundoff_tolerance':tol,'worker_startup_CPU':startup,'optimizer_steps':3072,'new_capability_fits':0,
        'torch':torch.__version__,'numpy':np.__version__,'platform':platform.platform(),
        'CPU':next(s.split(':',1)[1].strip() for s in Path('/proc/cpuinfo').read_text().splitlines() if s.startswith('model name')),
        'CPU_quota':Path('/sys/fs/cgroup/cpu.max').read_text().strip(),'threads':1,'concurrent_benchmark_workers':1}
    dump(OUT/'training.json',receipt);print('TRAINING SUMMARY',steady,total,receipt['status'],flush=True)


def frozen_audit():
    torch.set_num_threads(1);rows=source();receipts=[];values=physical_values=0
    for r in rows:
        key=r['key'];folder=OUT/'exports'/key;folder.mkdir(parents=True,exist_ok=True)
        z,meta=old.data(r['dataset']);tr=meta['split_edges'][0];x,_=old.batch(z,[tr,tr+24],r['horizon'])
        eager=build_model(r['horizon'],r['seed'],'eager').eval();compiled=build_model(r['horizon'],r['seed'],'scripted').eval()
        state=torch.load(ROOT/r['selected']['checkpoint'],weights_only=False)['state_dict'];eager.load_state_dict(state);compiled.load_state_dict(state)
        with torch.no_grad():a=eager(x).numpy();b=compiled(x).numpy()
        reference=np.load(ROOT/r['selected']['checkpoint'].rsplit('/',1)[0]/f"u{r['selected']['step']}-probe.npy")
        exact=bool(np.array_equal(a,reference) and np.array_equal(a,b));assert exact,key
        values+=a.size
        del eager.anchor_weight;del eager.anchor_bias;del compiled.anchor_weight;del compiled.anchor_bias
        ewrap=PhysicalForecast(eager,meta['mean'],meta['scale']).eval();cwrap=PhysicalForecast(compiled,meta['mean'],meta['scale']).eval()
        raw=x*cwrap.scale+cwrap.mean
        with torch.no_grad():ea=ewrap(raw).numpy();ca=cwrap(raw).numpy()
        assert np.array_equal(ea,ca),key+' physical'
        exported=torch.jit.trace(cwrap,raw,check_trace=False);path=folder/'physical.jit.pt';exported.save(str(path))
        with torch.no_grad():assert np.array_equal(exported(raw).numpy(),ea)
        np.save(folder/'input.npy',raw.numpy());np.save(folder/'output.npy',ea);physical_values+=ea.size
        receipts.append({'key':key,'lane':r['lane'],'dataset':r['dataset'],'horizon':r['horizon'],'seed':r['seed'],
            'source_checkpoint':r['selected'],'standardized_values_exact':a.size,'physical_values_exact':ea.size,
            'path':str(path.relative_to(ROOT)),'sha256':sha(path),'bytes':path.stat().st_size,'status':'PASS'})
        print('FROZEN',key,'PASS',flush=True)
    dump(OUT/'frozen-audit.json',{'status':'PASS','models':receipts,'standardized_values_exact':values,
        'physical_values_exact':physical_values,'no_new_test_scoring':True,'all_original_weights_unchanged':True})


def latency():
    torch.set_num_threads(1);rows=source();receipts=[]
    for r in rows:
        if r['lane']!='exposure' or r['dataset'] not in ('ETTh1','electricity') or r['horizon'] not in (96,720):continue
        z,meta=old.data(r['dataset']);x,_=old.batch(z,[96],r['horizon']);modes={}
        for backend in BACKENDS:
            t=time.process_time();m=build_model(r['horizon'],r['seed'],backend).eval()
            m.load_state_dict(torch.load(ROOT/r['selected']['checkpoint'],weights_only=False)['state_dict'])
            setup=time.process_time()-t;times=[]
            with torch.no_grad():
                for _ in range(5):m(x)
                for _ in range(30):
                    t=time.perf_counter();m(x);times.append((time.perf_counter()-t)*1000)
            modes[backend]={'setup_CPU_seconds':setup,'B1_median_ms':float(np.median(times)),'B1_p95_ms':float(np.quantile(times,.95)),'samples_ms':times}
        receipts.append({'case':r['key'],'modes':modes,'speedup':modes['eager']['B1_median_ms']/modes['scripted']['B1_median_ms']})
    dump(OUT/'latency.json',{'cases':receipts,'geometric_speedup':math.exp(np.mean([math.log(r['speedup']) for r in receipts])),
        'scope':'Eight same frozen native checkpoints, complete B1 standardized-input forward; core script vs eager, no concurrent local CPU-heavy work.'})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('train_benchmark','frozen_audit','latency'));args=p.parse_args();globals()[args.mode]()
