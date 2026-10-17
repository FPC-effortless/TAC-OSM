#!/usr/bin/env python3
"""DEV-036 paired CPU learning pilot and separate held-out development cohort."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
import torch
import torch.nn.functional as F
from run_local_mtsk_dev034 import build as original_build, training_cases
from run_learned_schema_dev import public_stream_tensors,public_queries,labels
from run_memory_controls_dev import physical,sensed
from tac_osm.development.bounded_mtsk import build_bounded
from tac_osm.development.delayed_maintenance import Action
from tac_osm.development.finite_clipping import clip_finite_cpu_gradients_
from tac_osm.development.noisy_memory import parameter_hash
from tac_osm.development.schema_maintenance import make_pair,ROLES,DEADLINES,execute,exact_public_targets
TRAINED=('unbounded_local_multi_fp64','bounded_local_multi_fp64')
REFERENCES=('mtsk_local_tied','masked_local','schema41')
ARMS=TRAINED+REFERENCES


def build(arm):
    if arm==TRAINED[0]:return original_build('mtsk_local_multi')
    if arm==TRAINED[1]:return build_bounded()
    if arm in REFERENCES:return original_build(arm)
    raise ValueError('unknown arm')


@torch.no_grad()
def evaluate(model,smoke=False):
    model.eval();rows=[];timing={'encode':0.,'decode':0.,'sensing':0.}
    conditions=([(32,'retention','at',0)] if smoke else
        [(delay,role,deadline,rep) for delay in (32,128,512) for role in ROLES for deadline in DEADLINES for rep in range(8)])
    for delay,role,deadline,rep in conditions:
        history=32 if smoke else 512
        cases=make_pair(910000+rep,history,delay,role,deadline,domain='dev036-v1-holdout')
        inputs=public_stream_tensors([c.observations for c in cases]);queries=public_queries([c.tick for c in cases],[c.horizon for c in cases])
        start=time.monotonic();state=model.encode(*inputs);timing['encode']+=time.monotonic()-start
        start=time.monotonic();logits=model.decode(state,queries);timing['decode']+=time.monotonic()-start
        probabilities=logits.softmax(-1)[...,1].tolist()
        start=time.monotonic();selected,matched=sensed(model,state,queries,cases,logits);timing['sensing']+=time.monotonic()-start
        for b,c in enumerate(cases):
            if exact_public_targets(c.observations,c.tick,c.horizon)!=c.wait_targets:raise AssertionError('public oracle mismatch')
            rows.append(dict(history=history,due_delay=delay,role=role,deadline=deadline,repetition=rep,partner=b,
                target_entity=c.target_entity,fingerprint=c.fingerprint,labels=labels([c])[0].tolist(),
                logits=logits[b].tolist(),predictions=logits[b].argmax(-1).tolist(),
                none=execute(c.world,c.horizon,physical(probabilities[b]),0),selected=selected[b],matched_random=matched[b],
                service_all=execute(c.world,c.horizon,tuple(range(8)),0)))
    mature=[r for r in rows if r['deadline']!='before'] if not smoke else rows
    cancel=[r for r in mature if r['role'] in ('pending_cancel','reissued_cancel')]
    primary=sum(r['predictions'][r['target_entity']][0]==r['labels'][r['target_entity']][0] for r in mature)/len(mature)
    cancellation=(sum(r['predictions'][r['target_entity']][0]==r['labels'][r['target_entity']][0] for r in cancel)/len(cancel)) if cancel else None
    all_entity=sum(r['predictions'][e][0]==r['labels'][e][0] for r in mature for e in range(8))/(8*len(mature))
    cross_entropy=F.cross_entropy(torch.tensor([r['logits'] for r in rows]).reshape(-1,2),torch.tensor([r['labels'] for r in rows]).reshape(-1)).item()
    physical_totals={}
    for policy in ('none','selected','matched_random','service_all'):
        physical_totals[policy]={k:sum(r[policy][k] for r in rows) for k in ('failures','unnecessary_services','inspections','total_cost')}
        physical_totals[policy]['services']=sum(len(r[policy]['services']) for r in rows)
    delta=(physical_totals['selected']['total_cost']-physical_totals['none']['total_cost'])/(8*len(rows))
    return dict(rows=rows,primary=primary,cancellation=cancellation,all_entity=all_entity,cross_entropy=cross_entropy,
        physical_totals=physical_totals,selected_minus_none_cost_per_entity=delta,timing_seconds=timing,
        mature_worlds=len(mature),cancellation_worlds=len(cancel))


def run(arm,output,pilot,smoke=False):
    torch.set_num_threads(1)
    if subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=ROOT,text=True).strip():raise ValueError('freeze source before measurement')
    source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    protocol=hashlib.sha256((ROOT/'research/dev036/protocol.json').read_bytes()).hexdigest()
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    if any((output/(arm+suffix)).exists() for suffix in ('.pt','.json','.failure.pt','.failure.json')):raise ValueError('preserve previous outcome')
    model=build(arm);initial=parameter_hash(model);start=time.monotonic();training_seconds=0.;curve=[];fps=[];upstream_hash=None
    if arm in REFERENCES:
        if smoke:raise ValueError('references require archived trained checkpoints')
        previous=json.loads((ROOT/'docs/DEV-034-PARTIAL-AUDIT.json').read_text())['arms'][arm]
        saved_path=Path(pilot)/(arm+'.pt');upstream_hash=hashlib.sha256(saved_path.read_bytes()).hexdigest()
        assert upstream_hash==previous['checkpoint_sha256']
        model.load_state_dict(torch.load(saved_path,map_location='cpu',weights_only=True)['model'])
    else:
        optimizer=torch.optim.AdamW(model.parameters(),lr=.002,weight_decay=.0001)
        steps=2 if smoke else 216
        try:
            for step in range(steps):
                cases=training_cases(step,smoke);fps.extend(c.fingerprint for c in cases)
                inputs=public_stream_tensors([c.observations for c in cases]);queries=public_queries([c.tick for c in cases],[c.horizon for c in cases]);targets=labels(cases)
                receipts=[(c.world.act(Action('inspect',random.Random(f'dev034:{step}:{b}').randrange(8))),) for b,c in enumerate(cases)]
                obs=public_stream_tensors(receipts);optimizer.zero_grad(set_to_none=True)
                state=model.encode(*inputs);pre=model.decode(state,queries);post=model.decode(model.update(state,*(x[:,0] for x in obs)),queries)
                loss=(F.cross_entropy(pre.reshape(-1,2),targets.reshape(-1))+F.cross_entropy(post.reshape(-1,2),targets.reshape(-1)))/2
                if not torch.isfinite(loss):raise FloatingPointError('nonfinite loss')
                loss.backward();norm=clip_finite_cpu_gradients_(model.parameters(),1.)
                optimizer.step()
                if step%27==0 or step==steps-1:
                    curve.append(dict(update=step+1,loss=float(loss.detach()),gradient_norm_float64=norm))
                    print(json.dumps(dict(arm=arm,**curve[-1])),flush=True)
        except Exception as error:
            failure=output/(arm+'.failure.pt')
            torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),rng=torch.get_rng_state(),attempted_update=step+1),failure)
            (output/(arm+'.failure.json')).write_text(json.dumps(dict(source=source,protocol_sha256=protocol,arm=arm,
                attempted_update=step+1,error=str(error),snapshot_sha256=hashlib.sha256(failure.read_bytes()).hexdigest()),indent=2)+'\n')
            raise
        training_seconds=time.monotonic()-start
    validation=evaluate(model,smoke)
    if set(fps)&{r['fingerprint'] for r in validation['rows']}:raise AssertionError('train/holdout overlap')
    checkpoint=output/(arm+'.pt');torch.save(dict(model=model.state_dict(),source=source,protocol_sha256=protocol,arm=arm,
        updates=(2 if smoke else 216) if arm in TRAINED else 0),checkpoint)
    report=dict(experiment='DEV-036',arm=arm,source=source,protocol_sha256=protocol,smoke=smoke,seed=311,
        initial_hash=initial,final_hash=parameter_hash(model),trained=arm in TRAINED,updates=(2 if smoke else 216) if arm in TRAINED else 0,
        fp64_clipping=arm in TRAINED,training_fingerprints=fps,checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        upstream_checkpoint_sha256=upstream_hash,parameters=sum(p.numel() for p in model.parameters()),state_scalars=model.state_scalars,
        training_seconds=training_seconds,elapsed_seconds=time.monotonic()-start,curve=curve,validation=validation,
        scientific_test_accessed=False,architecture_superiority_established=False,full_system_test6=False)
    (output/(arm+'.json')).write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(arm=arm,primary=validation['primary'],cancellation=validation['cancellation'],
        failures=validation['physical_totals']['selected']['failures'],cost=validation['physical_totals']['selected']['total_cost'],
        elapsed_seconds=report['elapsed_seconds'])),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--arm',choices=ARMS,required=True);p.add_argument('--output',required=True);p.add_argument('--pilot',required=True);p.add_argument('--smoke',action='store_true')
    a=p.parse_args();run(a.arm,a.output,a.pilot,a.smoke)
