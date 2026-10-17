#!/usr/bin/env python3
"""Frozen-memory simulator action supervision with equal-budget active/random data."""
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
from run_bounded_mtsk_dev036 import build
from run_learned_schema_dev import public_stream_tensors,public_queries
from tac_osm.development.schema_maintenance import make_pair,ROLES,DEADLINES,execute
from tac_osm.development.noisy_memory import parameter_hash
ARM='bounded_local_multi_fp64'


def teacher_action(wait,service):
    return int(1+4*service < 4*wait)


def choose(public,method):
    """Accept only public condition labels and frozen predictions, never worlds."""
    groups={}
    for i,row in enumerate(public):groups.setdefault((row['delay'],row['role'],row['deadline'],row['rep']),[]).append(i)
    assert len(groups)==288 and all(len(v)==2 for v in groups.values())
    rng=random.Random(37037);selected=[]
    for delay in (32,128,512):
        for role in ROLES:
            for deadline in DEADLINES:
                scores=[]
                for rep in range(8):
                    indices=groups[(delay,role,deadline,rep)]
                    margins=[]
                    for i in indices:
                        probs=public[i]['probabilities']
                        margins.append(min(abs(p[0]-p[1]-.25) for p in probs))
                    scores.append((sum(margins)/2,rep,indices))
                chosen=(sorted(scores)[:4] if method=='active' else rng.sample(scores,4))
                for _,_,indices in chosen:selected.extend(indices)
    assert len(selected)==288 and len(set(selected))==288
    return selected


def policy():
    torch.manual_seed(37037)
    return torch.nn.Sequential(torch.nn.Linear(3,16),torch.nn.Tanh(),torch.nn.Linear(16,1))


def train_policy(public,cases,indices):
    predictor=policy();features=torch.tensor([p for i in indices for p in public[i]['probabilities']],dtype=torch.float32)
    targets=torch.tensor([teacher_action(cases[i].wait_targets[e],cases[i].service_targets[e]) for i in indices for e in range(8)],dtype=torch.float32)
    optimizer=torch.optim.AdamW(predictor.parameters(),lr=.01,weight_decay=0.)
    start=time.monotonic();curve=[]
    for step in range(300):
        optimizer.zero_grad(set_to_none=True)
        loss=F.binary_cross_entropy_with_logits(predictor(features).squeeze(-1),targets,pos_weight=targets.new_tensor(3.))
        if not torch.isfinite(loss):raise FloatingPointError('nonfinite action loss')
        loss.backward();optimizer.step()
        if step in (0,99,199,299):curve.append(dict(update=step+1,loss=float(loss.detach())))
    return predictor,dict(training_seconds=time.monotonic()-start,curve=curve,teacher_positive_fraction=float(targets.mean()))


def choose_threshold(public,cases,indices):
    thresholds=(-.5,0.,.1,.2,.25,.3,.4,.5,.75)
    costs=[]
    for threshold in thresholds:
        total=0.
        for i in indices:
            case=cases[i];services=tuple(e for e,p in enumerate(public[i]['probabilities']) if p[0]-p[1]>threshold)
            total+=execute(case.world,case.horizon,services,0)['total_cost']
        costs.append(total)
    best=min(range(len(costs)),key=lambda k:costs[k])
    return thresholds[best],dict(thresholds=thresholds,random_training_costs=costs,selected_index=best)


@torch.no_grad()
def gather(model,domain,smoke=False):
    cases=[];public=[];inference=0.
    conditions=([(32,'retention','at',0)] if smoke else
        [(d,r,dl,rep) for d in (32,128,512) for r in ROLES for dl in DEADLINES for rep in range(8)])
    for delay,role,deadline,rep in conditions:
        episodes=make_pair(910000+rep,32 if smoke else 512,delay,role,deadline,domain=domain)
        inputs=public_stream_tensors([c.observations for c in episodes]);queries=public_queries([c.tick for c in episodes],[c.horizon for c in episodes])
        start=time.monotonic();p=model.decode(model.encode(*inputs),queries).softmax(-1)[...,1].tolist();inference+=time.monotonic()-start
        for partner,case in enumerate(episodes):
            cases.append(case)
            public.append(dict(delay=delay,role=role,deadline=deadline,rep=rep,partner=partner,fingerprint=case.fingerprint,
                target_entity=case.target_entity,probabilities=p[partner]))
    return cases,public,inference


@torch.no_grad()
def evaluate(public,cases,models,threshold):
    rows=[];totals={}
    for i,(item,case) in enumerate(zip(public,cases)):
        features=torch.tensor(item['probabilities'],dtype=torch.float32)
        actions={
            'fixed_threshold':tuple(e for e,p in enumerate(item['probabilities']) if p[0]-p[1]>.25),
            'tuned_threshold':tuple(e for e,p in enumerate(item['probabilities']) if p[0]-p[1]>threshold),
            'random_teacher':tuple(e for e,v in enumerate(models['random'](features).squeeze(-1)) if float(v)>=0),
            'active_teacher':tuple(e for e,v in enumerate(models['active'](features).squeeze(-1)) if float(v)>=0),
            'oracle':tuple(e for e in range(8) if teacher_action(case.wait_targets[e],case.service_targets[e]))}
        outcomes={name:execute(case.world,case.horizon,services,0) for name,services in actions.items()}
        rows.append(dict(**{k:v for k,v in item.items() if k!='probabilities'},probabilities=item['probabilities'],
            teacher_actions=list(actions['oracle']),outcomes=outcomes))
    for name in rows[0]['outcomes']:
        totals[name]={field:sum(r['outcomes'][name][field] for r in rows) for field in ('failures','unnecessary_services','inspections','total_cost')}
        totals[name]['services']=sum(len(r['outcomes'][name]['services']) for r in rows)
        totals[name]['teacher_agreement']=sum((e in r['outcomes'][name]['services'])==(e in r['teacher_actions']) for r in rows for e in range(8))/(len(rows)*8)
        subset=[r for r in rows if r['deadline']!='before' and r['role'] in ('pending_cancel','reissued_cancel')]
        totals[name]['mature_cancellation_target_agreement']=sum((r['target_entity'] in r['outcomes'][name]['services'])==(r['target_entity'] in r['teacher_actions']) for r in subset)/len(subset)
    return rows,totals


def run(input_dir,output,smoke=False):
    torch.set_num_threads(1);output=Path(output)
    if output.exists():raise ValueError('preserve earlier result')
    if subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=ROOT,text=True).strip():raise ValueError('freeze source before measurement')
    source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    protocol=hashlib.sha256((ROOT/'research/dev037/protocol.json').read_bytes()).hexdigest()
    prior=json.loads((ROOT/'docs/DEV-036-AUDITED-SUMMARY.json').read_text())['arms'][ARM]['checkpoint_sha256']
    checkpoint=Path(input_dir)/(ARM+'.pt')
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest()==prior
    model=build(ARM);model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True)['model']);model.eval()
    frozen_hash=parameter_hash(model);output.mkdir(parents=True)
    cases,pool,first_time=gather(model,'dev037-v1-pool',smoke)
    if smoke:
        # Smoke checks model-to-simulator boundaries, not full acquisition/training.
        assert len(pool)==2
        result={'smoke':True,'source':source,'protocol_sha256':protocol,'frozen_model_hash':frozen_hash,
            'probabilities_finite':bool(torch.isfinite(torch.tensor(pool[0]['probabilities'])).all()),
            'oracle_actions':[teacher_action(c.wait_targets[e],c.service_targets[e]) for c in cases for e in range(8)]}
        (output/'smoke.json').write_text(json.dumps(result,indent=2)+'\n');return
    random_indices=choose(pool,'random');active_indices=choose(pool,'active')
    assert len({x['fingerprint'] for x in pool})==len(pool)
    models={};training={}
    for name,indices in [('random',random_indices),('active',active_indices)]:
        models[name],training[name]=train_policy(pool,cases,indices)
        training[name]['selected_indices']=indices
        training[name]['selected_fingerprints']=[pool[i]['fingerprint'] for i in indices]
    threshold,threshold_tuning=choose_threshold(pool,cases,random_indices)
    evaluation_cases,evaluation_public,second_time=gather(model,'dev037-v1-eval')
    assert not {p['fingerprint'] for p in pool}&{p['fingerprint'] for p in evaluation_public}
    rows,totals=evaluate(evaluation_public,evaluation_cases,models,threshold)
    policies=output/'policies.pt';torch.save({name:m.state_dict() for name,m in models.items()},policies)
    report=dict(experiment='DEV-037',source=source,protocol_sha256=protocol,frozen_checkpoint_sha256=prior,
        frozen_model_hash=frozen_hash,policies_sha256=hashlib.sha256(policies.read_bytes()).hexdigest(),
        pool_public=pool,training=training,tuned_threshold=threshold,threshold_tuning=threshold_tuning,
        evaluation_rows=rows,evaluation_totals=totals,pool_inference_seconds=first_time,eval_inference_seconds=second_time,
        frozen_memory_optimizer_steps=0,full_system_test6=False,scientific_test_accessed=False,capability_established=False)
    (output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(totals=totals,tuned_threshold=threshold,training_seconds={k:v['training_seconds'] for k,v in training.items()})),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--output',required=True);p.add_argument('--smoke',action='store_true')
    a=p.parse_args();run(a.input,a.output,a.smoke)
