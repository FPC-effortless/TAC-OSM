#!/usr/bin/env python3
"""HISTORY-QUERY016: source-frozen, matched eight-clock internal research.

prepare: source-only split freeze and lr=0 full arithmetic inventory.
fit: requires committed preflight; every optimizer step must match its ledger.
replay: original source, final checkpoints and all scoring recomputed.
"""
from __future__ import annotations
import argparse
from collections import Counter
import datetime as dt
import json
import math
from pathlib import Path
import random
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from tac_osm.research_operator_budget import canonical,digest,verify_profile
from scripts.original_history_trace import sha_file,verify_implementation,load_original_modules,tensor_sha,verify_causal_features
from scripts.audit_content_query_specificity import predictions
from tac_osm.historical_query_metrics import window_scores,compare_errors
from scripts.validate_research_parity import check_manifest

DIR=ROOT/'research/mathematical-engineering/history_query016'
OUT=ROOT/'research/replay-bundles/HISTORY-QUERY016'
ARMS=('assoc_historic_query','fast_slow_historic_query')
SEEDS=(26016,26017)


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n')


def weight_hash(model):
    return digest({k:tensor_sha(v) for k,v in model.state_dict().items()})


def full_loss(model,d,rows,source):
    import torch
    from torch.nn import functional as F
    owner=torch.tensor([s for s,_ in rows],dtype=torch.long)
    anchor=torch.tensor([t for _,t in rows],dtype=torch.long)
    state=model.initial(len(rows))
    for offset in range(-23,65):
        t=anchor+offset
        state=model.step(state,*(d[k][owner,t] for k in ('values','valid','age','last')),owner)
    times=anchor[:,None]+torch.arange(0,64,8)[None,:]
    clocks=source.calendar(times.flatten().tolist()).reshape(len(rows),8,6)
    past=model.historic_event_logits(state,owner,clocks)
    future=source.calendar((anchor+70).tolist())[:,None,:]
    mu,sd=model.forecast(state,owner,future)
    # Targets first enter the computation after both predictions exist.
    mask=d['valid'][owner[:,None],times,0]
    truth=torch.where(mask,d['values'][owner[:,None],times,0],past.detach())
    historic=(F.smooth_l1_loss(past,truth,reduction='none')*mask).sum()/mask.sum().clamp_min(1)
    mu,sd=mu[:,0],sd[:,0]
    observed=d['valid'][owner,anchor+70]
    yt=torch.where(observed,d['values'][owner,anchor+70],mu.detach())
    nll=.5*(math.log(2*math.pi)+2*sd.log()+((yt-mu)/sd).square())
    forecast=((.8*nll+.2*(yt-mu).abs())*observed).sum()/observed.sum().clamp_min(1)
    return historic+forecast


def inventory_class():
    import torch
    from torch.utils._python_dispatch import TorchDispatchMode
    from torch.utils._pytree import tree_leaves
    class Inventory(TorchDispatchMode):
        def __init__(self):
            super().__init__()
            self.calls=Counter()
            self.stage='forward'
        def __torch_dispatch__(self,func,types,args=(),kwargs=None):
            kwargs=kwargs or {}
            result=func(*args,**kwargs)
            def shapes(x):
                return [(str(t.dtype),list(t.shape)) for t in tree_leaves(x) if isinstance(t,torch.Tensor)]
            signature=canonical([func._schema.name+'.'+func._schema.overload_name,shapes((args,kwargs)),shapes(result)]).decode()
            self.calls[self.stage,signature]+=1
            return result
        def receipt(self):
            return [dict(stage=s,signature=k,calls=n) for (s,k),n in sorted(self.calls.items())]
    return Inventory


def step(model,opt,d,rows,source):
    import torch
    audit=inventory_class()()
    model.train()
    opt.zero_grad(set_to_none=True)
    with audit:
        audit.stage='forward'
        loss=full_loss(model,d,rows,source)
        audit.stage='backward'
        loss.backward()
        audit.stage='clip'
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.,foreach=False)
        audit.stage='adamw'
        opt.step()
    if not torch.isfinite(loss) or any(p.grad is None or not torch.isfinite(p.grad).all() for p in model.parameters()):
        raise ValueError('nonfinite loss or missing/nonfinite active gradient')
    return float(loss.detach()),audit.receipt()


def setup(args):
    import torch
    recipe=json.loads((DIR/'RECIPE.json').read_text())
    verify_implementation(args.implementation_root,recipe['implementation_commit'])
    if sha_file(args.source)!=recipe['source_sha256']:
        raise ValueError('original source hash differs')
    source,original=load_original_modules(args.implementation_root)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    d=original.source_observations(args.source)
    verify_causal_features(d,source,source.hour(dt.datetime(2014,8,15))+70)
    return recipe,source,original,d


def rows_for(d,source,start,stop,every):
    result=[]
    for s in range(12):
        for t in range(source.hour(dt.datetime.fromisoformat(start)),source.hour(dt.datetime.fromisoformat(stop)),every):
            if bool(d['valid'][s,t,0]) and bool(d['valid'][s,t+70].any()):result.append([s,t])
    if len(result)<100 or len({s for s,_ in result})<10:raise ValueError('insufficient genuine source cohort')
    return result


def prepare(args):
    import torch
    recipe,source,original,d=setup(args)
    if (DIR/'FROZEN_PLAN.json').exists():raise ValueError('frozen preregistration already exists')
    train=rows_for(d,source,'2014-03-01','2014-04-18',12)
    primary=rows_for(d,source,'2014-07-01','2014-08-15',96)
    secondary=rows_for(d,source,'2014-05-01','2014-06-10',96)
    if max(t for _,t in train)+70>=min(t for _,t in secondary):raise ValueError('temporal split overlap')
    orders={}
    for seed in SEEDS:
        rng=random.Random(seed)
        orders[str(seed)]=[train[rng.randrange(len(train))] for _ in range(64*16)]
    plan=dict(recipe_sha256=digest(recipe),source_sha256=recipe['source_sha256'],training=train,
        primary=primary,secondary=secondary,orders=orders,
        all_cohorts_internally_exposed_FIT_not_virgin=True)
    args.output.mkdir(parents=True,exist_ok=True)
    write(args.output/'FROZEN_PLAN.json',plan)
    profile=dict(schema='tac-osm.complete-operator-profile.v1',recipe_sha256=digest(recipe),source_sha256=recipe['source_sha256'],profiles=[])
    # Real observations, derivative/optimizer workload; zero learning rate,
    # bitwise unchanged neural parameters. No fitted checkpoint generated.
    for seed in SEEDS:
        for arm in ARMS:
            model=original.create_model(seed,arm)
            if sum(p.numel() for p in model.parameters())!=99390:raise ValueError('model size changed')
            opt=torch.optim.AdamW(model.parameters(),lr=0.,weight_decay=.001,foreach=False,fused=False)
            for i,kind in enumerate(('first','warm')):
                before=weight_hash(model)
                loss,ops=step(model,opt,d,orders[str(seed)][i*16:(i+1)*16],source)
                profile['profiles'].append(dict(seed=seed,arm=arm,kind=kind,parameters=99390,
                    finite_gradient_tensors=sum(p.grad is not None for p in model.parameters()),
                    weights_before=before,weights_after=weight_hash(model),loss=loss,
                    operators=ops,operator_sha256=digest(ops)))
                print('PROFILE',seed,arm,kind,'signatures',len(ops),flush=True)
    write(args.output/'OPERATOR_PROFILE.json',profile)
    # Preserve exact inventory even if proof blocks. No silent fallback.
    proof=verify_profile(profile)
    write(args.output/'ARITHMETIC_PROOF.json',proof)
    manifest=json.loads((DIR/'MANIFEST_TEMPLATE.json').read_text())
    manifest.update(operator_profile_path=str((DIR/'OPERATOR_PROFILE.json').relative_to(ROOT)),
        operator_profile_sha256=sha_file(args.output/'OPERATOR_PROFILE.json'),training_recipe_sha256=digest(recipe))
    manifest['split_policy']['cohort_sha256']=digest(primary)
    contracts=dict(dataset_sha256=recipe['source_sha256'],train_split_sha256=digest(train),dev_split_sha256=digest(primary),
        test_split_sha256=digest('PROTECTED_TEST_UNOPENED_AND_EXCLUDED'),feature_contract_sha256=digest(recipe['causal_protocol']),
        target_contract_sha256=digest(recipe['objective']),optimizer_schedule_sha256=digest(recipe['optimizer']),
        tuning_policy_sha256=digest('NO_TUNING'),selection_policy_sha256=digest('FIXED_FINAL_64'),evaluation_contract_sha256=digest(recipe['evaluation']))
    manifest['arms']=[dict(seed=seed,architecture=arm,**contracts,training_sample_order_sha256=digest(orders[str(seed)]),
        trainable_parameters=99390,deployed_parameters=99390,nontrainable_state_bytes=6800,
        hardware_profile='CPU-FP32-single-thread',budget=dict(steps=64,examples_seen=1024,tokens_seen=90112,batch_size=16,
            history_events=88,queries_per_window=8,tuning_trials=0)) for seed in SEEDS for arm in ARMS]
    write(args.output/'FAIR_COMPARISON.json',manifest)
    print('PREFIT_FILES_READY_FOR_COMMIT; NO FIT EXECUTED',flush=True)


def frozen(args):
    import subprocess
    recipe,source,original,d=setup(args)
    for name in ('RECIPE.json','FROZEN_PLAN.json','OPERATOR_PROFILE.json','FAIR_COMPARISON.json'):
        path=DIR/name
        subprocess.run(['git','diff','--exit-code','HEAD','--',str(path)],cwd=ROOT,check=True,capture_output=True)
        subprocess.run(['git','ls-files','--error-unmatch',str(path)],cwd=ROOT,check=True,capture_output=True)
    p=json.loads((DIR/'FROZEN_PLAN.json').read_text())
    if p['recipe_sha256']!=digest(recipe) or p['source_sha256']!=recipe['source_sha256']:raise ValueError('frozen recipe/source changed')
    for key,start,stop,every in (('training','2014-03-01','2014-04-18',12),('primary','2014-07-01','2014-08-15',96),('secondary','2014-05-01','2014-06-10',96)):
        if p[key]!=rows_for(d,source,start,stop,every):raise ValueError('frozen source split differs')
    m=json.loads((DIR/'FAIR_COMPARISON.json').read_text())
    check_manifest(m,'preflight')
    return recipe,source,original,d,p,m


def evaluate(model,d,rows,source):
    import torch
    model.eval()
    with torch.no_grad():cases=predictions(model,d,rows,source)
    measured=[]
    for c,(s,t) in zip(cases,rows):
        times=list(range(t,t+64,8));mask=d['valid'][s,times,0].tolist()
        truth=[float(d['values'][s,z,0]) if ok else None for z,ok in zip(times,mask)]
        score=window_scores(c['predictions'],truth,mask)
        score['H64_zero_read_abs_error']=abs(c['zero_read_predictions'][0]-truth[0])
        c.update(targets_AFTER_prediction=truth,observed_AFTER_prediction=mask,scores=score)
        measured.append(score)
    keys=('H64_abs_error','all_query_MAE','collapsed_query_MAE','circular_wrong_query_MAE','H64_zero_read_abs_error')
    summary={k:sum(x[k] for x in measured)/len(measured) for k in keys}
    eligible=[x for x in measured if x['centered_MSE'] is not None]
    denom=sum(x['centered_null_MSE'] for x in eligible)
    summary.update(centered_skill=1-sum(x['centered_MSE'] for x in eligible)/denom if denom>0 else None,
        observed_targets=sum(x['observed_targets'] for x in measured),windows=len(rows))
    return dict(summary=summary,cases=cases)


def fit(args):
    import torch
    recipe,source,original,d,p,m=frozen(args)
    if OUT.exists():raise ValueError('new checkpoint directory exists; refuse overwrite')
    profile=json.loads((DIR/'OPERATOR_PROFILE.json').read_text())
    OUT.mkdir(parents=True)
    results=[]
    # No evaluation until every arm has passed postfit resource validation.
    for row in m['arms']:
        seed,arm=row['seed'],row['architecture'];samples=p['orders'][str(seed)]
        model=original.create_model(seed,arm)
        opt=torch.optim.AdamW(model.parameters(),lr=.0005,weight_decay=.001,foreach=False,fused=False)
        fingerprints=[];losses=[];start=time.monotonic()
        for i in range(64):
            value,ops=step(model,opt,d,samples[i*16:(i+1)*16],source)
            fingerprint=digest(ops)
            expected=next(r['operator_sha256'] for r in profile['profiles'] if r['seed']==seed and r['arm']==arm and r['kind']==('first' if i==0 else 'warm'))
            if fingerprint!=expected:raise ValueError(f'actual workload differs at step {i}; stop and preserve failure')
            fingerprints.append(fingerprint);losses.append(value)
            if i%16==0:print('FIT',seed,arm,'step',i+1,'loss',value,flush=True)
        path=OUT/f'seed-{seed}'/arm/'final.pt';path.parent.mkdir(parents=True)
        torch.save(dict(model=model.state_dict(),optimizer=opt.state_dict(),torch_rng=torch.get_rng_state(),
            python_rng=random.getstate(),seed=seed,architecture=arm,steps=64,examples_seen=1024,
            sample_order_sha256=digest(samples),recipe_sha256=digest(recipe),source_sha256=recipe['source_sha256'],
            training_script_sha256=sha_file(Path(__file__)),losses=losses,actual_operator_step_sha256=fingerprints),path)
        count=sum(t.numel() for t in model.parameters())
        row['measured']=dict(actual_steps=64,actual_sample_order_sha256=digest(samples),actual_operator_step_sha256=fingerprints,
            trainable_parameters=count,deployed_parameters=count,checkpoint_sha256=sha_file(path),cpu_elapsed_seconds=time.monotonic()-start)
        results.append(dict(seed=seed,arm=arm,checkpoint=str(path.relative_to(ROOT)),sha256=sha_file(path),losses=losses))
    write(OUT/'FINAL_COMPARISON.json',m)
    check_manifest(m,'final')
    for row in results:
        model=original.create_model(row['seed'],row['arm'])
        model.load_state_dict(torch.load(ROOT/row['checkpoint'],weights_only=False,map_location='cpu')['model'])
        row['primary']=evaluate(model,d,p['primary'],source)
        row['secondary']=evaluate(model,d,p['secondary'],source)
    comparisons=[]
    for seed in SEEDS:
        pair={r['arm']:r['primary'] for r in results if r['seed']==seed}
        a,b=pair[ARMS[0]],pair[ARMS[1]]
        stations=[s for s,_ in p['primary']]
        metrics=lambda q,k:[r['scores'][k] for r in q['cases']]
        comp={control:compare_errors(stations,metrics(a,control),metrics(a,'all_query_MAE')) for control in ('collapsed_query_MAE','circular_wrong_query_MAE')}
        comp['fastslow_all_query_MAE']=compare_errors(stations,metrics(b,'all_query_MAE'),metrics(a,'all_query_MAE'))
        comp['zero_read_H64']=compare_errors(stations,metrics(a,'H64_zero_read_abs_error'),metrics(a,'H64_abs_error'))
        skill=a['summary']['centered_skill']
        comparisons.append(dict(seed=seed,comparisons=comp,gate_pass=skill is not None and skill>0 and all(v['gate_pass'] for v in comp.values())))
    write(OUT/'RESULTS.json',dict(evidence_level='internally_measured_not_independently_qualified',models=results,
        comparisons=comparisons,all_seed_gate_pass=all(x['gate_pass'] for x in comparisons),
        protected_TEST_opened=False,source_sha256=recipe['source_sha256'],recipe_sha256=digest(recipe),PNDS_full_graph_IEEE_AD='OPEN'))


def replay(args):
    import torch
    recipe,source,original,d,p,m=frozen(args)
    m=json.loads((OUT/'FINAL_COMPARISON.json').read_text());check_manifest(m,'final')
    results=json.loads((OUT/'RESULTS.json').read_text())
    for row in results['models']:
        path=ROOT/row['checkpoint']
        if sha_file(path)!=row['sha256']:raise ValueError('checkpoint bytes changed')
        c=torch.load(path,map_location='cpu',weights_only=False)
        if c['recipe_sha256']!=digest(recipe) or c['sample_order_sha256']!=digest(p['orders'][str(row['seed'])]) or c['training_script_sha256']!=sha_file(Path(__file__)):
            raise ValueError('source/order/script checkpoint binding differs')
        model=original.create_model(row['seed'],row['arm']);model.load_state_dict(c['model'],strict=True)
        for cohort in ('primary','secondary'):
            if digest(evaluate(model,d,p[cohort],source))!=digest(row[cohort]):raise ValueError('fresh source/checkpoint replay differs')
        print('REPLAY PASS',row['seed'],row['arm'],flush=True)
    write(args.output/'REPLAY_RECEIPT.json',dict(status='PASS_EXACT_SOURCE_CHECKPOINT_REPLAY',results_sha256=sha_file(OUT/'RESULTS.json'),checkpoints=[r['sha256'] for r in results['models']]))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=('prepare','fit','replay'),required=True)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--implementation-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    {'prepare':prepare,'fit':fit,'replay':replay}[args.mode](args)
if __name__=='__main__':main()
