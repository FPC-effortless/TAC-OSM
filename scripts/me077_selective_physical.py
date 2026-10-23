"""ME077 real-data capability study. Fit, freeze all seeds in Git, then score.

No TEST opened, no new policy/architecture superiority claim. Reuses original
ME076 source/split/query definitions, never its optimizer or model bytes.
"""
from __future__ import annotations
import argparse, copy, hashlib, json, math, random, subprocess, time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from scripts.me076_train_system1_real import (
    load_original, valid_origins, build_batch, evaluation_origins, sha, write,
    digest, TIME_STEP_SECONDS, MOTION_PHYSICAL_THRESHOLD)
from scripts.validate_research_parity import check_manifest
from tac_osm.parameter_budget import budgeted_optimizer, checked_optimizer_step, save_training_checkpoint, validate_runtime_evidence
from tac_osm.variants.selective_physical import SelectivePhysicalModel, mixture_nll, mixture_quantile

ROOT=Path('research/mathematical-engineering/me077')
OUT=Path('research/replay-bundles/ME077')
MANIFEST=ROOT/'FAIR_COMPARISON.json'
SEEDS=tuple(range(77001,77006))
UPDATES=1024
BATCH=24
COUNT=102503

def integrity():
    m=json.loads(MANIFEST.read_text())
    if check_manifest(m,'preflight')['status']!='PASS_PREFIT_SINGLE_MODEL':
        raise RuntimeError('preregistration rejected')
    for path,expected in m['implementation_sha256'].items():
        if sha(path)!=expected: raise RuntimeError('frozen implementation changed: '+path)
    return m

def make_model(identified):
    return SelectivePhysicalModel(identified['delta_scale'], np.maximum(4*identified['servo_step_limit'], .1))

def rollout(model,x,a):
    h=None
    tick=x.new_full((len(x),),TIME_STEP_SECONDS)
    for j in range(x.shape[1]):
        velocity=torch.zeros_like(x[:,j]) if j==0 else x[:,j]-x[:,j-1]
        pred=model(x[:,j],a[:,j],velocity,tick,h)
        h=pred.fast
    return pred

def order(fit,identified,seed):
    pools=valid_origins(fit,identified)
    tasks=sorted({k[0] for k in pools})
    flat={task:[] for task in tasks}
    for (task,ep),(times,moving) in sorted(pools.items()):
        flat[task].extend((task,ep,t) for t in times)
    rng=np.random.default_rng(seed)
    samples=[[flat[tasks[(step*BATCH+b)%len(tasks)]][int(rng.integers(len(flat[tasks[(step*BATCH+b)%len(tasks)]])))] for b in range(BATCH)] for step in range(UPDATES)]
    return samples, {'sampling':'equal-task, uniform eligible origins, NO outcome oversampling',
       'order_sha256':digest(samples),'supervised_targets':UPDATES*BATCH,
       'source_eligible_windows':sum(len(v) for v in flat.values())}

def qualify(cache):
    integrity()
    _,fit,report,identified,evidence=load_original(cache)
    queries,stats=evaluation_origins(report,identified)
    n=sum(p.numel() for p in make_model(identified).parameters())
    if n!=COUNT: raise RuntimeError('live model count altered')
    evidence.update(stats)
    evidence.update({'actual_trainable_parameters':n,'actual_deployed_parameters':n,
      'point_estimator':'median of stationary/moving Laplace mixture',
      'REPORT_previously_exposed':True,'protected_tissue_box_opened':False})
    write(ROOT/'SOURCE_QUALIFICATION.json',evidence)
    return fit,report,identified,evidence,queries

def fit_all(cache):
    integrity()
    if list(OUT.glob('seed-*/final.pt')): raise RuntimeError('NEVER overwrite original trained models')
    fit,report,identified,evidence,queries=qualify(cache)
    records=[]
    for seed in SEEDS:
        torch.manual_seed(seed);np.random.seed(seed);random.seed(seed)
        model=make_model(identified)
        folder=OUT/f'seed-{seed}'
        samples,stats=order(fit,identified,seed)
        opt=budgeted_optimizer(model,torch.optim.AdamW,model.parameters(),lr=.0007,
            weight_decay=.0001,budget_manifest=MANIFEST,budget_seed=seed,
            budget_architecture=model.architecture,budget_receipt=folder/'live-budget.json')
        trace=[];start=time.process_time()
        for step,picks in enumerate(samples,1):
            x,a,y=build_batch(fit,picks)
            pred=rollout(model,x,a)
            actual_motion=(((y-x[:,-1]).abs()*torch.as_tensor(identified['state_scale'])).mean(-1)>=MOTION_PHYSICAL_THRESHOLD).float()
            loss=mixture_nll(pred,y)+.2*F.binary_cross_entropy_with_logits(pred.motion_logit,actual_motion)
            if not torch.isfinite(loss): raise RuntimeError('nonfinite real loss')
            opt.zero_grad(set_to_none=True);loss.backward()
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            if not torch.isfinite(norm): raise RuntimeError('nonfinite real gradient')
            checked_optimizer_step(opt)
            if step==1 or step%128==0:
                row={'seed':seed,'step':step,'FIT_loss':float(loss.detach())}
                trace.append(row); print(json.dumps(row),flush=True)
        checkpoint=folder/'final.pt'
        save_training_checkpoint({'weights':model.state_dict(),'optimizer':opt.state_dict(),
           'seed':seed,'updates':UPDATES,'torch_rng':torch.get_rng_state(),
           'numpy_rng':(np.random.get_state()[0],np.random.get_state()[1].tolist(),*np.random.get_state()[2:]),'python_rng':random.getstate(),
           'FIT_only_identified':{k:v.tolist() for k,v in identified.items()},
           'source':evidence,'sampling':stats,'source_registration_sha256':sha(MANIFEST)},checkpoint)
        rec={'seed':seed,'parameters':COUNT,'actual_steps':UPDATES,'training_cpu_seconds':time.process_time()-start,
             'sampling':stats,'checkpoint_sha256':sha(checkpoint),'trace':trace}
        write(folder/'FIT_RECEIPT.json',rec);records.append(rec)
    write(OUT/'FIT_INDEX.json',records)
    print('ME077_FITS_COMPLETE_NO_REPORT_SCORED',flush=True)

def frozen():
    for seed in SEEDS:
        folder=OUT/f'seed-{seed}'
        for name in ('final.pt','live-budget.json','FIT_RECEIPT.json'):
            path=folder/name
            committed=subprocess.check_output(['git','show','HEAD:'+str(path)])
            if hashlib.sha256(committed).hexdigest()!=sha(path):
                raise RuntimeError('all original models must be Git-frozen before REPORT scoring')

def score_model(model,report,identified,queries):
    model.eval();rows=[]
    physical=torch.as_tensor(identified['state_scale'])
    with torch.no_grad():
        for i in range(0,len(queries),24):
            picks=queries[i:i+24];x,a,y=build_batch(report,picks)
            pred=rollout(model,x,a)
            median=mixture_quantile(pred,.5)
            lower=mixture_quantile(pred,.05);upper=mixture_quantile(pred,.95)
            error=((median-y).abs()*physical).mean(-1)
            pers=((x[:,-1]-y).abs()*physical).mean(-1)
            pm=pred.motion_logit.sigmoid()
            truth=pers>=MOTION_PHYSICAL_THRESHOLD
            p=pred.mixture_logit.sigmoid()
            d=y-pred.current
            ll0=F.logsigmoid(-pred.mixture_logit)-(2*pred.stationary_scale).log()-d.abs()/pred.stationary_scale
            ll1=F.logsigmoid(pred.mixture_logit)-(2*pred.moving_scale).log()-(d-pred.moving_delta).abs()/pred.moving_scale
            nll=-torch.logaddexp(ll0,ll1).mean(-1)
            width=(upper-lower)*physical
            iscore=width+20*(lower-y).clamp_min(0)*physical+20*(y-upper).clamp_min(0)*physical
            for j,(task,ep,t) in enumerate(picks):
                rows.append({'task':task,'episode_index':ep,'frame_index':t,
                 'physical_mae_model':float(error[j]),'physical_mae_persistence':float(pers[j]),
                 'motion_rich':bool(truth[j]),'motion_probability':float(pm[j]),
                 'motion_brier':float((pm[j]-truth[j].float()).square()),
                 'normalized_mixture_nll':float(nll[j]),
                 'physical_interval_width':float(width[j].mean()),
                 'physical_interval_score':float(iscore[j].mean()),
                 'coordinate_interval_coverage':float(((lower[j]<=y[j])&(y[j]<=upper[j])).float().mean()),
                 'target_normalized':y[j].tolist(),'median_normalized':median[j].tolist(),
                 'lower_normalized':lower[j].tolist(),'upper_normalized':upper[j].tolist(),
                 'mixture_probability':p[j].tolist()})
    return rows

def summarize(rows):
    keys=('physical_mae_model','physical_mae_persistence','motion_brier','normalized_mixture_nll',
          'physical_interval_width','physical_interval_score','coordinate_interval_coverage')
    return {'n':len(rows),**{k:float(np.mean([r[k] for r in rows])) for k in keys}} if rows else {'n':0}

def run_scores(cache,replay=False):
    integrity();frozen()
    _,_,report,identified,evidence=load_original(cache)
    queries,_=evaluation_origins(report,identified)
    allrows=[];records=[]
    for seed in SEEDS:
        folder=OUT/f'seed-{seed}';fit=json.loads((folder/'FIT_RECEIPT.json').read_text())
        if sha(folder/'final.pt')!=fit['checkpoint_sha256']:raise RuntimeError('checkpoint SHA changed')
        model=make_model(identified)
        state=torch.load(folder/'final.pt',map_location='cpu',weights_only=True)
        model.load_state_dict(state['weights'],strict=True)
        if state['updates']!=UPDATES:raise RuntimeError('actual update count differs')
        rows=score_model(model,report,identified,queries)
        metrics={name:summarize(rs) for name,rs in (
          ('all',rows),('movement_rich',[r for r in rows if r['motion_rich']]),
          ('near_stationary',[r for r in rows if not r['motion_rich']]))}
        oldpath=Path('research/replay-bundles/ME076')/f'seed-{76000+seed%1000}'/'ORIGINAL_REAL_PREDICTIONS.json'
        old=json.loads(oldpath.read_text())
        if len(rows)!=len(old):raise RuntimeError('old/new query denominator differs')
        for row,prior in zip(rows,old):
            if (row['task'],row['episode_index'],row['frame_index'])!=(prior['task'],prior['episode_index'],prior['frame_index']):raise RuntimeError('old/new query identity differs')
            if row['target_normalized']!=prior['measured_next_state_normalized']:raise RuntimeError('old/new source target differs')
            row['prior_ME076_physical_mae']=prior['physical_mae_model']
            row['physical_mae_servo']=prior['physical_mae_fit_servo_lag1']
        path=folder/'ORIGINAL_PREDICTIONS.json'
        if replay:
            original=json.loads(path.read_text())
            if len(original)!=len(rows):raise RuntimeError('replay denominator changed')
            for a,b in zip(original,rows):
                for k in a:
                    if isinstance(a[k],(float,list)):
                        if not np.allclose(a[k],b[k],rtol=1e-5,atol=1e-6):raise RuntimeError('raw replay failed: '+k)
                    elif a[k]!=b[k]:raise RuntimeError('case replay failed: '+k)
        else:
            if path.exists():raise RuntimeError('never overwrite original predictions')
            write(path,rows);write(folder/'RESULTS.json',metrics)
        allrows.append(rows);records.append({'seed':seed,'metrics':metrics,'checkpoint_sha256':sha(folder/'final.pt')})
    if replay:
        write(OUT/'INDEPENDENT_REPLAY.json',{'status':'PASS','original_models':5,
          'original_real_queries_per_seed':len(queries),'raw_prediction_rtol':1e-5,'raw_prediction_atol':1e-6,
          'no_refit':True,'source_sha256':evidence['original_160_episode_source_sha256']})
    else:
        write(OUT/'ORIGINAL_SUMMARY.json',{'study':'ME077','seeds':records,'source':evidence,
          'architecture_superiority':False,'independent_generalization':False})
    return records

def finalize():
    integrity();frozen()
    final=copy.deepcopy(json.loads(MANIFEST.read_text()))
    final['prefit_registration_sha256']=sha(MANIFEST)
    for row in final['models']:
        folder=OUT/f"seed-{row['seed']}";checkpoint=folder/'final.pt';receipt=folder/'live-budget.json'
        live=json.loads(receipt.read_text())
        if live['successful_optimizer_steps']!=UPDATES:raise RuntimeError('actual step count wrong')
        row['measured']={'trainable_parameters':COUNT,'deployed_parameters':COUNT,'actual_steps':UPDATES,
          'checkpoint_sha256':sha(checkpoint),'runtime_evidence':{'receipt_path':str(receipt),'receipt_sha256':sha(receipt),
          'checkpoint_path':str(checkpoint),'checkpoint_sha256':sha(checkpoint)}}
    result=check_manifest(final,'final');write(ROOT/'FINAL_COMPARISON.json',final);print(result)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['qualify','fit','score','replay','finalize']);p.add_argument('--cache',type=Path,default=Path('/tmp/me077-original-source'));args=p.parse_args()
    torch.set_num_threads(1)
    if args.mode=='qualify':qualify(args.cache)
    elif args.mode=='fit':fit_all(args.cache)
    elif args.mode=='score':run_scores(args.cache)
    elif args.mode=='replay':run_scores(args.cache,True)
    else:finalize()
