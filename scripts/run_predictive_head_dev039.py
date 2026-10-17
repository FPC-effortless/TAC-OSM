#!/usr/bin/env python3
"""DEV-039 frozen-memory all-role predictive-head refit and physical decisions."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
import torch
import torch.nn.functional as F
from run_bounded_mtsk_dev036 import build
from run_learned_schema_dev import public_stream_tensors,public_queries,labels
from run_memory_controls_dev import physical
from tac_osm.development.schema_maintenance import make_pair,ROLES,DEADLINES,execute
from tac_osm.development.noisy_memory import parameter_hash
ARM='bounded_local_multi_fp64'


@torch.no_grad()
def encoded(model,domain):
    cases=[];features=[];target=[];conditions=[];duration=0.
    for delay in (32,128,512):
        for role in ROLES:
            for deadline in DEADLINES:
                for rep in range(8):
                    episode=make_pair(930000+rep,512,delay,role,deadline,domain=domain)
                    events=public_stream_tensors([c.observations for c in episode]);queries=public_queries([c.tick for c in episode],[c.horizon for c in episode])
                    start=time.monotonic();state=model.encode(*events);q=model.query_encoder(queries)
                    read=model.adapter(model.memory.read(state,q));x=torch.cat((read,q),-1)
                    assert torch.equal(model.head(x),model.decode(state,queries))
                    duration+=time.monotonic()-start
                    for b,c in enumerate(episode):
                        cases.append(c);features.append(x[b].detach().clone());target.append(labels([c])[0])
                        conditions.append(dict(delay=delay,role=role,deadline=deadline,rep=rep,partner=b,
                            fingerprint=c.fingerprint,target_entity=c.target_entity))
    return cases,torch.stack(features),torch.stack(target),conditions,duration


def fit(head,x,y,lr):
    optimizer=torch.optim.AdamW(head.parameters(),lr=lr,weight_decay=.0001)
    flat=x.reshape(-1,44);targets=y.reshape(-1)
    start=time.monotonic();curve=[]
    for step in range(500):
        optimizer.zero_grad(set_to_none=True);logits=head(flat)
        loss=F.cross_entropy(logits,targets)
        if not torch.isfinite(loss):raise FloatingPointError('nonfinite refit loss')
        loss.backward();optimizer.step()
        if step in (0,99,199,299,399,499):curve.append(dict(update=step+1,loss=float(loss.detach())))
    return dict(seconds=time.monotonic()-start,curve=curve)


@torch.no_grad()
def evaluate(head,x,y,cases,conditions):
    logits=head(x);probabilities=logits.softmax(-1)[...,1].tolist();pred=logits.argmax(-1).tolist();rows=[]
    for b,(case,condition) in enumerate(zip(cases,conditions)):
        services=physical(probabilities[b]);outcome=execute(case.world,case.horizon,services,0)
        rows.append(dict(**condition,labels=y[b].tolist(),logits=logits[b].tolist(),predictions=pred[b],
            services=list(services),physical=outcome))
    mature=[r for r in rows if r['deadline']!='before'];cancel=[r for r in mature if r['role'] in ('pending_cancel','reissued_cancel')];reissued=[r for r in mature if r['role']=='reissued_cancel']
    accuracy=lambda cohort:sum(r['predictions'][r['target_entity']][0]==r['labels'][r['target_entity']][0] for r in cohort)/len(cohort)
    physical_totals={field:sum(r['physical'][field] for r in rows) for field in ('failures','unnecessary_services','inspections','total_cost')}
    physical_totals['services']=sum(len(r['services']) for r in rows)
    return dict(rows=rows,primary=accuracy(mature),cancellation=accuracy(cancel),reissued=accuracy(reissued),
        all_entity=sum(r['predictions'][e][0]==r['labels'][e][0] for r in mature for e in range(8))/(len(mature)*8),
        physical_totals=physical_totals,cross_entropy=float(F.cross_entropy(logits.reshape(-1,2),y.reshape(-1))))


def run(input_dir,output):
    torch.set_num_threads(1);output=Path(output)
    if output.exists():raise ValueError('preserve prior result')
    if subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=ROOT,text=True).strip():raise ValueError('freeze source before measurement')
    source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    protocol=hashlib.sha256((ROOT/'research/dev039/protocol.json').read_bytes()).hexdigest()
    expected=json.loads((ROOT/'docs/DEV-036-AUDITED-SUMMARY.json').read_text())['arms'][ARM]['checkpoint_sha256']
    checkpoint=Path(input_dir)/(ARM+'.pt');assert hashlib.sha256(checkpoint.read_bytes()).hexdigest()==expected
    model=build(ARM);model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True)['model']);model.eval()
    model_hash=parameter_hash(model);train_cases,train_x,train_y,train_conditions,train_time=encoded(model,'dev039-v1-train')
    assert len(train_cases)==576 and len({c.fingerprint for c in train_cases})==576
    heads={'original':model.head,'refit_mlp':torch.nn.Sequential(torch.nn.Linear(44,64),torch.nn.Tanh(),torch.nn.Linear(64,2))}
    heads['refit_mlp'].load_state_dict(model.head.state_dict())
    torch.manual_seed(39039);heads['linear']=torch.nn.Linear(44,2)
    training={'refit_mlp':fit(heads['refit_mlp'],train_x,train_y,.002),'linear':fit(heads['linear'],train_x,train_y,.02)}
    eval_cases,eval_x,eval_y,eval_conditions,eval_time=encoded(model,'dev039-v1-eval')
    assert len(eval_cases)==576 and not {c.fingerprint for c in train_cases}&{c.fingerprint for c in eval_cases}
    results={name:evaluate(head,eval_x,eval_y,eval_cases,eval_conditions) for name,head in heads.items()}
    output.mkdir(parents=True)
    raw=output/'features-heads.pt';torch.save(dict(train_features=train_x,train_labels=train_y,eval_features=eval_x,eval_labels=eval_y,
        original=model.head.state_dict(),refit_mlp=heads['refit_mlp'].state_dict(),linear=heads['linear'].state_dict()),raw)
    report=dict(experiment='DEV-039',source=source,protocol_sha256=protocol,checkpoint_sha256=expected,model_hash=model_hash,
        raw_sha256=hashlib.sha256(raw.read_bytes()).hexdigest(),train_conditions=train_conditions,eval_conditions=eval_conditions,
        training=training,results=results,train_encoding_seconds=train_time,eval_encoding_seconds=eval_time,
        frozen_encoder_memory_steps=0,head_refit_steps=500,scientific_test_accessed=False,full_system_capability_established=False)
    (output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({name:{k:v[k] for k in ('primary','cancellation','reissued','physical_totals')} for name,v in results.items()}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();run(a.input,a.output)
