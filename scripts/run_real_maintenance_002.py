"""Frozen baseline-first fitting; sealed B has no load path in this program."""
import argparse
import copy
from concurrent.futures import ProcessPoolExecutor,as_completed
import json
import multiprocessing
import resource
from pathlib import Path
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import numpy as np
import torch
from tac_osm.development.residual_forecast import ResidualForecast,specifications
from data_real_maintenance_002 import prepare,write,digest,open_a
PROTOCOL=ROOT/'research/real-maint-002/PROTOCOL.md'


def make(spec,seed,normalization):
    m=ResidualForecast(**spec,seed=seed);m.location.copy_(normalization['location']);m.scale.copy_(normalization['scale']);return m


def loss(m,x,y,naive_scale):return ((m(x)-y).abs().mean(0)/naive_scale.where(naive_scale>1e-6,torch.ones_like(naive_scale))).mean()


def relative(mae,baseline):
    return [float(a/b) if b>0 else 1. if a==0 else None for a,b in zip(mae,baseline,strict=True)]


@torch.no_grad()
def evaluate(m,data,naive_scale,rows=False):
    dtype=next(m.parameters()).dtype;errors=[];squared=[];baseline=[]
    for j in range(0,len(data['x']),256):
        x=data['x'][j:j+256].to(dtype);y=data['y'][j:j+256].to(dtype);prediction=m(x);assert torch.isfinite(prediction).all();errors.append((prediction-y).abs());squared.append(((prediction-y)/m.scale)**2);baseline.append((x[:,-1]-y).abs())
    e=torch.cat(errors);b=torch.cat(baseline);mae=e.mean(0).tolist();naive=b.mean(0).tolist();ratios=relative(mae,naive);scales=naive_scale.tolist();mase=[a/s if s>1e-6 else None for a,s in zip(mae,scales,strict=True)];finite=all(v is not None for v in ratios);result={'raw_MAE':mae,'persistence_MAE':naive,'MASE':mase,'relative_MAE':ratios,'mean_MASE':float(np.mean(mase)) if all(v is not None for v in mase) else None,'worst_relative_MAE':max(ratios) if finite else None,'all_channels_no_worse':finite and max(ratios)<=1.,'context_normalized_MSE':torch.cat(squared).mean(0).tolist(),'windows':len(e)}
    if rows:result.update(absolute_error=e.tolist(),persistence_absolute_error=b.tolist(),time=data['time'].tolist())
    return result


def score(v,step):return (float('inf') if v['worst_relative_MAE'] is None else v['worst_relative_MAE'],float('inf') if v['mean_MASE'] is None else v['mean_MASE'],step)


def group(job):
    seed,lrs,specs,folder,data_path,smoke=job;torch.set_num_threads(1);folder=Path(folder);folder.mkdir();start=time.process_time();data=Path(data_path);train=torch.load(data/'train.pt',weights_only=True);val=torch.load(data/'validation.pt',weights_only=True);norm={'location':train['x'].mean((0,1)),'scale':train['x'].std((0,1)).clamp_min(1e-5)};naive=(train['y']-train['x'][:,-1]).abs().mean(0);models={a:make(s,seed,norm) for a,s in specs.items()};ema={a:copy.deepcopy(m).eval() for a,m in models.items()};opt={a:torch.optim.AdamW(m.parameters(),lr=lrs[a],weight_decay=1e-4) for a,m in models.items()};curves={a:[{'update':0,'validation':evaluate(copy.deepcopy(m).double(),val,naive.double()),'eligible_for_selection':False}] for a,m in ema.items()};best={};rng=torch.Generator().manual_seed(seed);status='RUNNING';steps=8 if smoke else 512
    try:
        for step in range(1,steps+1):
            ids=torch.randint(len(train['x']),(64,),generator=rng)
            for a,m in models.items():
                value=loss(m,train['x'][ids],train['y'][ids],naive);assert torch.isfinite(value);opt[a].zero_grad();value.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1.,error_if_nonfinite=True);opt[a].step()
                with torch.no_grad():
                    for e,p in zip(ema[a].parameters(),m.parameters(),strict=True):e.mul_(.99).add_(p,alpha=.01)
                if step%64==0 or step==steps:
                    v=evaluate(copy.deepcopy(ema[a]).double(),val,naive.double());curves[a].append({'update':step,'train_loss':float(value.detach()),'validation':v,'eligible_for_selection':True})
                    if a not in best or score(v,step)<best[a][0]:best[a]=(score(v,step),copy.deepcopy(ema[a]).double().eval(),step)
            if time.process_time()-start>30:status='CAPPED';break
        else:status='COMPLETE'
    except Exception:
        import traceback
        status='FAILED';write(folder/'failure.json',{'trace':traceback.format_exc()})
    reports=[]
    for a,m in models.items():
        path=folder/a;path.mkdir();torch.save(m.state_dict(),path/'raw.pt');torch.save(ema[a].state_dict(),path/'ema-final.pt');selected=best[a][1] if a in best else copy.deepcopy(ema[a]).double();chosen=best[a][2] if a in best else 0;torch.save(selected.state_dict(),path/'agent.pt');r={'arm':a,'seed':seed,'status':status,'updates':step,'selected_update':chosen,'spec':specs[a],'learning_rate':lrs[a],'curves':curves[a],'validation':evaluate(selected,val,naive),'naive_scale':naive.tolist(),'checkpoint_sha256':digest(path/'agent.pt'),'source':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()};write(path/'report.json',r);reports.append(r)
    if time.process_time()-start>30 and status=='COMPLETE':status='CAPPED'
    result={'seed':seed,'status':'SMOKE_ONLY' if smoke else status,'cpu_seconds':time.process_time()-start,'reports':reports};write(folder/'summary.json',result);return result


def run(directory,source,smoke=False):
    torch.set_num_threads(1);start=time.process_time();child_start=resource.getrusage(resource.RUSAGE_CHILDREN);root=Path(directory);root.mkdir(parents=True,exist_ok=False);specs=specifications();data=root/'_datasets';offset=10000000*int(smoke);groups=[];fits=[]
    if smoke:
        data.mkdir();rng=torch.Generator().manual_seed(1930001)
        for n in ('train','validation'):
            x=torch.randn(64,24,8,generator=rng);torch.save({'x':x,'y':x[:,-1]+.1*torch.randn(64,8,generator=rng),'time':torch.arange(64)},data/(n+'.pt'))
    else:
        assert not subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=ROOT,text=True).strip(),'Commit frozen source and registration first';prepare(source,data)
    preparation=time.process_time()-start
    with ProcessPoolExecutor(max_workers=2,mp_context=multiprocessing.get_context('spawn')) as pool:
        jobs=[(93000+offset,{a:lr for a in specs},specs,str(root/f'pilot-{i}'),str(data),smoke) for i,lr in enumerate((.0003,) if smoke else (.0003,.001))]
        pilots=[f.result() for f in as_completed([pool.submit(group,j) for j in jobs])];groups+=pilots;write(root/'pilots.json',pilots)
        lrs={a:min([r for g in pilots for r in g['reports'] if r['arm']==a],key=lambda r:(*score(r['validation'],r['selected_update']),r['learning_rate']))['learning_rate'] for a in specs};write(root/'learning-rates.json',lrs)
        if all(g['status']==('SMOKE_ONLY' if smoke else 'COMPLETE') for g in pilots):
            seeds=[93001+offset] if smoke else range(93001,93021)
            for begin in range(0,len(seeds),2):
                selected=list(seeds)[begin:begin+2]
                if preparation+sum(g['cpu_seconds'] for g in groups)+len(selected)*30>1200:break
                jobs=[(seed,lrs,specs,str(root/f'group-{seed}'),str(data),smoke) for seed in selected]
                for future in as_completed([pool.submit(group,j) for j in jobs]):
                    g=future.result();groups.append(g);fits.append(g);write(root/'groups.json',fits);print('GROUP',g['seed'],g['status'],round(g['cpu_seconds'],2),flush=True)
    child_end=resource.getrusage(resource.RUSAGE_CHILDREN);fit_cpu=time.process_time()-start+child_end.ru_utime+child_end.ru_stime-child_start.ru_utime-child_start.ru_stime
    complete=fit_cpu<=1200 and len(fits)==(1 if smoke else 20) and all(g['status']==('SMOKE_ONLY' if smoke else 'COMPLETE') for g in fits);summary={'status':'SMOKE_ONLY' if smoke else 'FITS_COMPLETE' if complete else 'INCOMPLETE','fits':len(fits)*4,'pilot_fits':len(pilots)*4,'fitting_cpu_seconds':fit_cpu,'protocol_sha256':digest(PROTOCOL),'test_A_opened':False,'test_B_opened':False};write(root/'summary.json',summary)
    if complete and not smoke:
        by_arm={a:[r for g in fits for r in g['reports'] if r['arm']==a] for a in specs}
        arm=min(specs,key=lambda a:(-sum(r['validation']['all_channels_no_worse'] for r in by_arm[a]),float(np.median([score(r['validation'],r['selected_update'])[0] for r in by_arm[a]])),a));hashes={str(p.relative_to(root)):digest(p) for p in sorted(root.glob('group-*/*/agent.pt'))};write(root/'selected-arm.json',{'arm':arm,'rule':'validation seed-pass count, median worst channel ratio, lexical arm','checkpoint_hashes':hashes,'source':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()});test=open_a(data,root)
        for g in fits:
            for r in g['reports']:
                path=root/f"group-{r['seed']}"/r['arm'];m=ResidualForecast(**r['spec'],seed=r['seed']).double().eval();m.load_state_dict(torch.load(path/'agent.pt',weights_only=True));write(path/'test-a.json',evaluate(m,test,torch.tensor(r['naive_scale'],dtype=torch.float64),True))
        summary.update(status='COMPLETE',test_A_opened=True);write(root/'summary.json',summary)
    print('SUMMARY',json.dumps(summary),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--directory',required=True);p.add_argument('--source');p.add_argument('--smoke',action='store_true');a=p.parse_args();run(a.directory,a.source,a.smoke)
