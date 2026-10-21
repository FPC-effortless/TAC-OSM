#!/usr/bin/env python3
"""ME044 complete real forecasting models; historically inspected DEV only."""
import argparse,hashlib,json,platform,resource,subprocess,time
from pathlib import Path
import numpy as np
import torch
from scripts.me043_train_adaptive import DOMAINS,save,sha
from tac_osm.me043_adaptive_compute import AdaptiveForecaster
from tac_osm.me044_multiscale_decoder import ARMS,MATCHED,MultiscaleForecaster,build_model,objective,clip_gradients,is_expert

SEEDS=(44001,44002);HORIZONS=(96,720);LANES=('exposure','cpu')
UPDATES=1000;CPU_SECONDS=10.;MAX_UPDATES=10000;BATCH=4
torch.set_num_threads(1)


def period(domain):return 144 if domain=='tetouan' else 24


def predict(net,x):
    if isinstance(net,MultiscaleForecaster):
        cheap,_,residual,_,trust,_=net.training_outputs(x,True)
        return cheap+residual,trust,cheap
    if isinstance(net,AdaptiveForecaster):
        p,q=net.routed(x);return p,q,None
    return net(x),None,None


@torch.no_grad()
def evaluate(net,z,mod,h,path=None):
    net.eval();origins=np.arange(mod.TR,mod.VA-h+1,48,dtype=np.int64)
    ps=[];ys=[];qs=[];cs=[]
    for j in range(0,len(origins),16):
        x,y=mod.samples(z,origins[j:j+16],h)
        p,q,c=predict(net,x)
        if not bool(torch.isfinite(p).all()):raise ArithmeticError('nonfinite original forecast')
        ps.append(p.numpy());ys.append(y.numpy())
        if q is not None:qs.append(q.numpy())
        if c is not None:cs.append(c.numpy())
    p=np.concatenate(ps);y=np.concatenate(ys)
    # Fix array order before reduction; archive reloads preserve C ordering.
    d=np.ascontiguousarray(p,dtype=np.float64)-np.ascontiguousarray(y,dtype=np.float64)
    mse=(d*d).mean((0,1));mae=np.abs(d).mean((0,1));start=-3 if z.shape[1]==8 else -1
    row={'primary_MSE':float(mse[start:].mean()),'channel_MSE':mse.tolist(),
         'primary_MAE':float(mae[start:].mean()),'channel_MAE':mae.tolist(),
         'all_channel_MSE':float(mse.mean()),'windows':len(origins),'forecast_scalars':int(p.size)}
    q=np.concatenate(qs) if qs else np.empty((0,1),np.float32)
    cheap=np.concatenate(cs) if cs else np.empty((0,),np.float32)
    if q.size:row['mean_gate']=float(q.mean())
    if isinstance(net,AdaptiveForecaster):row['native_fraction']=float((q>.5).mean())
    if cheap.size:
        e=np.ascontiguousarray(cheap,dtype=np.float64)-np.ascontiguousarray(y,dtype=np.float64)
        row['cheap_primary_MSE']=float((e*e).mean((0,1))[start:].mean())
    if path:
        np.savez_compressed(path,pred=p,truth=y,origin=origins,q=q,cheap=cheap)
        row['predictions_sha256']=sha(path)
    return row


def fit(z,mod,domain,h,seed,arm,lane,parent,prec,out):
    folder=out/f'{lane}-h{h}-s{seed}-{arm}';folder.mkdir()
    net=build_model(arm,h,seed,period(domain));net.initialize_ar(parent)
    opt=torch.optim.AdamW([{'params':list(net.ar.parameters()),'lr':.0001},
        {'params':[p for n,p in net.named_parameters() if not n.startswith('ar.')],'lr':.001}],weight_decay=1e-4)
    sampler=torch.Generator().manual_seed(seed+2000)
    cpu=0.;trace=[];elapsed_start=time.perf_counter();process_start=time.process_time()
    ceiling=UPDATES if lane=='exposure' else MAX_UPDATES
    for step in range(1,ceiling+1):
        anchors=torch.randint(96,mod.TR-h+1,(BATCH,),generator=sampler)
        x,y=mod.samples(z,anchors,h)
        start=time.process_time();net.train();opt.zero_grad(set_to_none=True)
        loss,mse=objective(net,x,y,step)
        if not bool(torch.isfinite(loss)):raise ArithmeticError('nonfinite genuine training objective')
        loss.backward();grad=clip_gradients(net)
        if not bool(torch.isfinite(grad)):raise ArithmeticError('nonfinite original gradient')
        opt.step();cpu+=time.process_time()-start
        trace.append({'step':step,'loss':float(loss.detach()),'MSE':float(mse.detach()),'optimizer_CPU':cpu})
        if lane=='cpu' and cpu>=CPU_SECONDS:break
    fit_elapsed=time.perf_counter()-elapsed_start;fit_process=time.process_time()-process_start
    metric=evaluate(net,z,mod,h,folder/'DEV.npz')
    checkpoint={'state_dict':net.state_dict(),'optimizer':opt.state_dict(),
        'torch_rng':torch.get_rng_state(),'sampler_rng':sampler.get_state(),
        'arm':arm,'horizon':h,'seed':seed,'domain':domain,'period':period(domain),
        'lane':lane,'updates':step,'source_sha256':mod.CSV_SHA,
        'training_end':mod.TR,'data_numeric_end':mod.VA}
    torch.save(checkpoint,folder/'final.pt')
    torch.save({k:v for k,v in checkpoint.items() if k not in ('optimizer','torch_rng','sampler_rng')},folder/'inference.pt')
    save(folder/'trace.json',trace)
    row={'arm':arm,'horizon':h,'seed':seed,'lane':lane,'updates':step,
        'expert_updates':step//4 if arm in MATCHED else step//8 if arm=='previous_scheduled' else 0,
        'examples':step*BATCH,'target_scalars':step*BATCH*h*z.shape[1],
        'parameter_count':sum(p.numel() for p in net.parameters()),
        'optimizer_CPU':cpu,'training_process_CPU':fit_process,'training_elapsed_wall':fit_elapsed,
        'fit_and_DEV_process_CPU':time.process_time()-process_start,
        'parent_total_CPU':prec['total_CPU_including_DEV'],'parent_optimizer_CPU':prec['optimizer_CPU_seconds'],
        'peak_process_RSS_KiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        'metric':metric,'checkpoint':str((folder/'final.pt').relative_to(out)),
        'checkpoint_sha256':sha(folder/'final.pt'),'inference_sha256':sha(folder/'inference.pt'),
        'CPU_cap_reached':cpu>=CPU_SECONDS if lane=='cpu' else None}
    save(folder/'summary.json',row)
    print(json.dumps({'status':'ME044_FIT','domain':domain,'horizon':h,'seed':seed,'arm':arm,
        'lane':lane,'updates':step,'CPU':cpu,'MSE':metric['primary_MSE']}),flush=True)
    return row


def summarize(rows):
    out={}
    for lane in LANES:
        out[lane]={}
        for arm in ARMS:
            r=[v for v in rows if v['lane']==lane and v['arm']==arm];assert len(r)==4
            out[lane][arm]={'primary_MSE':float(np.mean([v['metric']['primary_MSE'] for v in r])),
                'CPU':float(np.mean([v['optimizer_CPU'] for v in r])),
                'updates':float(np.mean([v['updates'] for v in r])),
                'by_horizon':{str(h):float(np.mean([v['metric']['primary_MSE'] for v in r if v['horizon']==h])) for h in HORIZONS}}
    rivals=('pyramid_context','pyramid_gru','original_context','previous_scheduled')
    ex=out['exposure'];cp=out['cpu'];candidate=ex['pyramid_native']
    out['native_gates']={
        'one_percent_accuracy':candidate['primary_MSE']<=.99*min(ex[a]['primary_MSE'] for a in rivals),
        'no_large_horizon_regression':all(candidate['by_horizon'][str(h)]<=1.05*min(ex[a]['by_horizon'][str(h)] for a in rivals) for h in HORIZONS),
        'equal_CPU_accuracy':cp['pyramid_native']['primary_MSE']<=min(cp[a]['primary_MSE'] for a in rivals)}
    out['native_gates']['all']=all(out['native_gates'].values())
    return out


def run(out):
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    report={'study':'ME044','source_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'completed':False,'domains':{},'new_test_numeric_values':0,'architectural_superiority_claim':False,
        'environment':{'python':platform.python_version(),'torch':torch.__version__,'numpy':np.__version__,
        'threads':torch.get_num_threads(),'processor':platform.processor(),'platform':platform.platform()}}
    for domain,mod in DOMAINS.items():
        dest=out/domain;dest.mkdir();z,provenance=mod.get_source(dest)
        data={'provenance':provenance,'parents':[],'fits':[]};report['domains'][domain]=data
        for h in HORIZONS:
            for seed in SEEDS:
                parent,prec=mod.fit_parent(z,h,seed,dest);data['parents'].append(prec)
                for lane in LANES:
                    for arm in ARMS:
                        data['fits'].append(fit(z,mod,domain,h,seed,arm,lane,parent,prec,dest))
                        save(out/'results.json',report)
        data['evaluation']=summarize(data['fits']);save(out/'results.json',report)
    report['completed']=True;save(out/'results.json',report)


@torch.no_grad()
def audit(out,portable_reference=None):
    out=Path(out);report=json.loads((out/'results.json').read_text());assert report['completed']
    values=0;models=0;optimizer_count=0;cheap_retention=0
    portable=json.loads((Path(portable_reference)/'records.json').read_text()) if portable_reference else None
    for domain,data in report['domains'].items():
        mod=DOMAINS[domain];dest=out/domain;z,_=mod.get_source(dest)
        for row in data['fits']:
            cp=dest/row['checkpoint'];assert sha(cp)==row['checkpoint_sha256']
            ck=torch.load(cp,weights_only=True,map_location='cpu')
            net=build_model(row['arm'],row['horizon'],row['seed'],period(domain));net.load_state_dict(ck['state_dict']);net.eval()
            assert sha(cp.parent/'inference.pt')==row['inference_sha256']
            names=[n for n,p in net.named_parameters() if n.startswith('ar.')]
            names +=[n for n,p in net.named_parameters() if not n.startswith('ar.')]
            ids=[i for g in ck['optimizer']['param_groups'] for i in g['params']]
            assert len(ck['optimizer']['state'])==len(names)==len(ids)
            for n,i in zip(names,ids):
                updates=row['updates']
                if row['arm'] in MATCHED and is_expert(n):updates//=4
                if row['arm']=='previous_scheduled' and (n.startswith(('memory.','router.')) or n=='native_gain'):updates//=8
                assert int(ck['optimizer']['state'][i]['step'])==updates,(n,updates)
            optimizer_count+=1
            path=cp.parent/'DEV.npz';assert sha(path)==row['metric']['predictions_sha256']
            metric=row['metric']
            if portable is not None:
                item=next(v for v in portable['rows'] if v['domain']==domain and all(v[k]==row[k] for k in ('arm','lane','horizon','seed')))
                assert item['checkpoint_sha256']==row['checkpoint_sha256']
                path=Path(portable_reference)/item['reference'];assert sha(path)==item['sha256'];metric=item['metric']
            with np.load(path) as saved:
                ps=[];ys=[];qs=[];cs=[];origins=saved['origin']
                for j in range(0,len(origins),16):
                    x,y=mod.samples(z,origins[j:j+16],row['horizon']);p,q,c=predict(net,x)
                    ps.append(p.numpy());ys.append(y.numpy())
                    if q is not None:qs.append(q.numpy())
                    if c is not None:cs.append(c.numpy())
                pred=np.concatenate(ps);truth=np.concatenate(ys)
                assert np.array_equal(pred,saved['pred']),str(cp)
                assert np.array_equal(truth,saved['truth'])
                if qs:assert np.array_equal(np.concatenate(qs),saved['q'])
                if cs:assert np.array_equal(np.concatenate(cs),saved['cheap'])
                delta=np.ascontiguousarray(pred,dtype=np.float64)-np.ascontiguousarray(truth,dtype=np.float64)
                mse=(delta*delta).mean((0,1));mae=np.abs(delta).mean((0,1))
                # Canonical C reductions; strict equality primary; independent
                # epsilon/count bound is also recorded for alternate libraries.
                n=pred.shape[0]*pred.shape[1];u=np.finfo(np.float64).eps/2
                gamma=(2*n+8)*u/(1-(2*n+8)*u)
                assert np.all(np.abs(mse-metric['channel_MSE'])<=2*gamma*np.maximum(mse,np.finfo(float).tiny))
                assert np.all(np.abs(mae-metric['channel_MAE'])<=2*gamma*np.maximum(mae,np.finfo(float).tiny))
                values+=pred.size;models+=1
            if row['arm'] in MATCHED and row['lane']=='exposure':
                ref=next(r for r in data['fits'] if r['arm']=='original_context' and r['lane']=='exposure' and r['horizon']==row['horizon'] and r['seed']==row['seed'])
                old=torch.load(dest/ref['checkpoint'],weights_only=True)['state_dict']
                assert all(torch.equal(ck['state_dict'][k],v) for k,v in old.items())
                cheap_retention+=1
    result={'status':'PASS','models':models,'optimizer_states':optimizer_count,
        'forecast_scalars_bit_exact':values,'cheap_trajectory_checks_exact':cheap_retention,
        'truth_and_gate_arrays_exact':True,'MSE_MAE_recomputed':True,'new_fits':0,'new_test_rows':0}
    result['forecast_reference']='separate_AVX2_serving' if portable is not None else 'original_AVX512'
    if portable is not None:result['original_AVX512_cross_ISA_replay']='FAILED_RETAINED'
    save(out/('portable-audit.json' if portable is not None else 'audit.json'),result);print(json.dumps(result),flush=True)


@torch.no_grad()
def latency(out):
    out=Path(out);report=json.loads((out/'results.json').read_text());rows=[]
    for domain,data in report['domains'].items():
        mod=DOMAINS[domain];z,_=mod.get_source(out/domain)
        for row in data['fits']:
            cp=out/domain/row['checkpoint'];assert sha(cp)==row['checkpoint_sha256']
            net=build_model(row['arm'],row['horizon'],row['seed'],period(domain))
            net.load_state_dict(torch.load(cp,weights_only=True)['state_dict']);net.eval()
            x,_=mod.samples(z,[mod.TR],row['horizon'])
            for _ in range(5):net(x)
            timings=[]
            for _ in range(5):
                start=time.process_time()
                for __ in range(20):net(x)
                timings.append((time.process_time()-start)/20)
            rows.append({k:row[k] for k in ('arm','horizon','seed','lane')}|{
                'domain':domain,'batch':1,'channels':z.shape[1],
                'median_CPU_seconds':float(np.median(timings)),'repetitions':timings})
    save(out/'latency.json',{'rows':rows,'full_normalization_AR_context_and_expert_included':True})
    print('ME044_LATENCY_PASS',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',required=True)
    p.add_argument('--portable-reference');p.add_argument('--audit',action='store_true');p.add_argument('--latency',action='store_true');a=p.parse_args()
    if a.audit:audit(a.out,a.portable_reference)
    elif a.latency:latency(a.out)
    else:run(a.out)
