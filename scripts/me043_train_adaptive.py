#!/usr/bin/env python3
"""ME043 full-model real-data development. No new TEST labels accessed."""
import argparse,hashlib,json,platform,resource,time
from pathlib import Path
import numpy as np
import torch
from scripts import me041_train_architecture_portfolio as power
from scripts import me042_real_transformer_crossdomain as oil
from tac_osm.me036_patch_forecast import build_model as ar_model
from tac_osm.me043_adaptive_compute import ARMS,COSTS,AdaptiveForecaster,build_model,objective

SEEDS=(43001,43002);HORIZONS=(96,720);LANES=('exposure','cpu')
UPDATES=250;CPU_SECONDS=5.;MAX_UPDATES=4000;BATCH=4
DOMAINS={'tetouan':power,'etth1':oil}
torch.set_num_threads(1)

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(v,indent=2,sort_keys=True)+'\n')

@torch.no_grad()
def evaluate(net,z,mod,h,path=None):
    net.eval();origins=np.arange(mod.TR,mod.VA-h+1,48,dtype=np.int64)
    predictions=[];targets=[];routing=[]
    for j in range(0,len(origins),16):
        x,y=mod.samples(z,origins[j:j+16],h)
        if isinstance(net,AdaptiveForecaster):
            p,q=net.routed(x);routing.append(q.numpy())
        else:p=net(x)
        if not bool(torch.isfinite(p).all()):raise ArithmeticError('nonfinite real forecast')
        predictions.append(p.numpy());targets.append(y.numpy())
    pred=np.concatenate(predictions);truth=np.concatenate(targets)
    diff=pred.astype(np.float64)-truth.astype(np.float64)
    mse=np.mean(diff**2,axis=(0,1));mae=np.mean(np.abs(diff),axis=(0,1))
    start=-3 if mod is power else -1
    metric={'primary_MSE':float(mse[start:].mean()),'all_channel_MSE':float(mse.mean()),
            'channel_MSE':mse.tolist(),'channel_MAE':mae.tolist(),
            'primary_MAE':float(mae[start:].mean()),'windows':len(origins),
            'forecast_scalars':int(pred.size),'native_fraction':None}
    q=np.concatenate(routing) if routing else np.empty((0,1),np.float32)
    if routing:
        metric['native_fraction']=float((q>.5).mean())
        metric['mean_native_probability']=float(q.mean())
    if path is not None:
        np.savez_compressed(path,pred=pred,truth=truth,origin=origins,q=q)
        metric['predictions_sha256']=sha(path)
    return metric


def train_case(z,mod,h,seed,arm,lane,parent,prec,out):
    dest=out/f'{lane}-h{h}-s{seed}-{arm}';dest.mkdir()
    net=build_model(arm,h,seed);net.initialize_ar(parent)
    sampler=torch.Generator().manual_seed(seed+2000)
    opt=torch.optim.AdamW([
        {'params':list(net.ar.parameters()),'lr':.0001},
        {'params':[p for n,p in net.named_parameters() if not n.startswith('ar.')],'lr':.001}
    ],weight_decay=1e-4)
    trace=[];cpu=0.;start=time.process_time()
    cap=UPDATES if lane=='exposure' else MAX_UPDATES
    for step in range(1,cap+1):
        anchors=torch.randint(96,mod.TR-h+1,(BATCH,),generator=sampler)
        x,y=mod.samples(z,anchors,h)
        t=time.process_time();net.train();opt.zero_grad(set_to_none=True)
        loss,mse=objective(net,x,y)
        if not bool(torch.isfinite(loss)):raise ArithmeticError('nonfinite FIT objective')
        loss.backward();grad=torch.nn.utils.clip_grad_norm_(net.parameters(),1.)
        if not bool(torch.isfinite(grad)):raise ArithmeticError('nonfinite FIT gradient')
        opt.step();cpu+=time.process_time()-t
        trace.append({'step':step,'MSE':float(mse.detach()),'loss':float(loss.detach()),'CPU':cpu})
        if lane=='cpu' and cpu>=CPU_SECONDS:break
    metric=evaluate(net,z,mod,h,dest/'DEV.npz')
    checkpoint={'state_dict':net.state_dict(),'optimizer':opt.state_dict(),
        'torch_rng':torch.get_rng_state(),'sampler_rng':sampler.get_state(),
        'arm':arm,'horizon':h,'seed':seed,'lane':lane,'updates':step,
        'source_sha256':mod.CSV_SHA,'data_numeric_end':mod.VA,'training_end':mod.TR}
    torch.save(checkpoint,dest/'final.pt');save(dest/'trace.json',trace)
    # Original inference state also preserved separately for portable delivery.
    torch.save({k:v for k,v in checkpoint.items() if k not in
        ('optimizer','torch_rng','sampler_rng')},dest/'inference.pt')
    row={'arm':arm,'horizon':h,'seed':seed,'lane':lane,'updates':step,
         'examples':step*BATCH,'target_scalars':step*BATCH*h*z.shape[1],
         'parameter_count':sum(p.numel() for p in net.parameters()),
         'optimizer_CPU':cpu,'total_training_and_DEV_CPU':time.process_time()-start,
         'parent_total_CPU':prec['total_CPU_including_DEV'],
         'parent_optimizer_CPU':prec['optimizer_CPU_seconds'],
         'peak_RSS_KiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
         'checkpoint':str((dest/'final.pt').relative_to(out)),
         'checkpoint_sha256':sha(dest/'final.pt'),
         'inference_sha256':sha(dest/'inference.pt'),'metric':metric,
         'CPU_cap_reached':cpu>=CPU_SECONDS if lane=='cpu' else None}
    save(dest/'summary.json',row)
    print(json.dumps({'status':'ME043_FIT','domain':out.name,**{k:row[k] for k in
        ('arm','horizon','seed','lane','updates','optimizer_CPU')},'MSE':metric['primary_MSE'],
        'native_fraction':metric['native_fraction']}),flush=True)
    return row


def summarize(rows):
    result={}
    for lane in LANES:
        scores={}
        for arm in ARMS:
            r=[v for v in rows if v['lane']==lane and v['arm']==arm]
            assert len(r)==4
            scores[arm]={'primary_MSE':float(np.mean([v['metric']['primary_MSE'] for v in r])),
                'CPU':float(np.mean([v['optimizer_CPU'] for v in r])),
                'updates':float(np.mean([v['updates'] for v in r])),
                'by_horizon':{str(h):float(np.mean([v['metric']['primary_MSE'] for v in r
                    if v['horizon']==h])) for h in HORIZONS},
                'native_fraction':float(np.mean([v['metric']['native_fraction'] for v in r]))
                    if arm in COSTS else None}
        result[lane]=scores
    result['gates']={}
    for arm in COSTS:
        exp=result['exposure'];cpu=result['cpu'];v=exp[arm]
        acc=v['primary_MSE']<=.99*min(exp['context_cross']['primary_MSE'],exp['innovation_tac']['primary_MSE'])
        horizon=all(v['by_horizon'][str(h)]<=1.05*min(exp['context_cross']['by_horizon'][str(h)],
                     exp['innovation_tac']['by_horizon'][str(h)]) for h in HORIZONS)
        practical=cpu[arm]['primary_MSE']<=min(cpu['context_cross']['primary_MSE'],cpu['innovation_tac']['primary_MSE'])
        result['gates'][arm]={'one_percent_accuracy':bool(acc),'no_large_horizon_regression':bool(horizon),
                               'equal_CPU_accuracy':bool(practical),'all':bool(acc and horizon and practical)}
    return result


def run(out):
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    report={'study':'ME043','source_commit':__import__('subprocess').check_output(
        ['git','rev-parse','HEAD'],text=True).strip(),'environment':{'torch':torch.__version__,
        'numpy':np.__version__,'python':platform.python_version(),'machine':platform.machine(),
        'CPU_threads':torch.get_num_threads()},'domains':{},'completed':False,
        'new_test_numeric_values_opened':0,'architectural_superiority_claim':False}
    for domain,mod in DOMAINS.items():
        dest=out/domain;dest.mkdir()
        z,provenance=mod.get_source(dest)
        data={'provenance':provenance,'fits':[],'parents':[]}
        report['domains'][domain]=data
        for h in HORIZONS:
            for seed in SEEDS:
                state,prec=mod.fit_parent(z,h,seed,dest);data['parents'].append(prec)
                for lane in LANES:
                    for arm in ARMS:
                        data['fits'].append(train_case(z,mod,h,seed,arm,lane,state,prec,dest))
                        save(out/'results.json',report)
        data['evaluation']=summarize(data['fits'])
        save(out/'results.json',report)
    report['completed']=True;save(out/'results.json',report)


@torch.no_grad()
def replay(out):
    out=Path(out);report=json.loads((out/'results.json').read_text());assert report['completed']
    values=0;models=0
    for domain,data in report['domains'].items():
        mod=DOMAINS[domain];dest=out/domain;z,_=mod.get_source(dest)
        for row in data['fits']:
            cp=dest/row['checkpoint'];assert sha(cp)==row['checkpoint_sha256']
            c=torch.load(cp,weights_only=True,map_location='cpu')
            net=build_model(row['arm'],row['horizon'],row['seed'])
            net.load_state_dict(c['state_dict'],strict=True);net.eval()
            path=cp.parent/'DEV.npz';assert sha(path)==row['metric']['predictions_sha256']
            with np.load(path) as saved:
                origins=saved['origin'];predictions=[];truth=[];qs=[]
                for j in range(0,len(origins),16):
                    x,y=mod.samples(z,origins[j:j+16],row['horizon'])
                    if isinstance(net,AdaptiveForecaster):p,q=net.routed(x);qs.append(q.numpy())
                    else:p=net(x)
                    predictions.append(p.numpy());truth.append(y.numpy())
                actual=np.concatenate(predictions)
                assert np.array_equal(actual,saved['pred']),str(cp)
                assert np.array_equal(np.concatenate(truth),saved['truth'])
                if qs:assert np.array_equal(np.concatenate(qs),saved['q'])
                values+=actual.size;models+=1
    receipt={'status':'PASS','models':models,'forecast_scalars_bit_exact':values,
             'truth_and_router_arrays_exact':True,'new_fits':0,'new_test_values':0}
    save(out/'replay.json',receipt);print(json.dumps(receipt),flush=True)


@torch.no_grad()
def latency(out):
    out=Path(out);report=json.loads((out/'results.json').read_text());assert report['completed']
    rows=[]
    for domain,data in report['domains'].items():
        mod=DOMAINS[domain];dest=out/domain;z,_=mod.get_source(dest)
        for row in data['fits']:
            # All selected original models; no validation-driven checkpoint choice.
            cp=dest/row['checkpoint'];assert sha(cp)==row['checkpoint_sha256']
            net=build_model(row['arm'],row['horizon'],row['seed'])
            net.load_state_dict(torch.load(cp,weights_only=True)['state_dict']);net.eval()
            x,_=mod.samples(z,[mod.TR],row['horizon'])
            for _ in range(5):net(x)
            timings=[]
            for _ in range(5):
                start=time.process_time()
                for k in range(20):net(x)
                timings.append((time.process_time()-start)/20)
            rows.append({'domain':domain,'arm':row['arm'],'seed':row['seed'],
                'lane':row['lane'],'horizon':row['horizon'],'batch':1,'channels':z.shape[1],
                'median_CPU_seconds':float(np.median(timings)),'repetitions':timings})
    save(out/'latency.json',{'rows':rows,'hardware':platform.processor(),
                           'includes_normalization_AR_router_and_native_when_selected':True})
    print('ME043_REAL_INPUT_LATENCY_COMPLETE',flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True)
    ap.add_argument('--replay',action='store_true');ap.add_argument('--latency',action='store_true')
    a=ap.parse_args()
    if a.replay:replay(a.out)
    elif a.latency:latency(a.out)
    else:run(a.out)
