#!/usr/bin/env python3
"""Frozen DEV-034 factorial engineering pilot, validation only."""
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
from tac_osm.development.local_mtsk import ARMS as FACTORIAL,LocalMTSKPredictor
from tac_osm.development.entity_aging import EntityAgingControl
from tac_osm.development.memory_baselines import MemoryBaseline
from tac_osm.development.schema_maintenance import make_pair,ROLES,DEADLINES,execute,exact_public_targets
from tac_osm.development.delayed_maintenance import Action
from tac_osm.development.noisy_memory import parameter_hash
from run_learned_schema_dev import public_stream_tensors,public_queries,labels
from run_memory_controls_dev import physical,sensed
ARMS=FACTORIAL+('masked_local','schema41')


def build(arm):
    if arm in FACTORIAL:return LocalMTSKPredictor(arm,34311)
    if arm=='masked_local':return EntityAgingControl(arm,34311)
    if arm=='schema41':return MemoryBaseline(arm,34311)
    raise ValueError('unregistered arm')


def training_cases(step,smoke):
    return make_pair(311000000+step,32 if smoke else 512,(32,128,512)[(step//4)%3],
        ROLES[step%4],DEADLINES[(step//12)%3],domain='dev034-v1-train')


@torch.no_grad()
def evaluate(model,smoke):
    model.eval();rows=[];timing=dict(encode=0.,decode=0.,sensing=0.)
    for history in ((32,) if smoke else (128,512)):
        for delay in ((32,) if smoke else (32,128,512)):
            for role in ROLES:
                for deadline in (('at',) if smoke else DEADLINES):
                    for rep in range(1 if smoke else 2):
                        cases=make_pair(900000+rep,history,delay,role,deadline,domain='dev034-v1-validation')
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
                                none=execute(c.world,c.horizon,physical(probabilities[b]),0),selected=selected[b],matched_random=matched[b]))
    primary={};cancellation={}
    for history in sorted({r['history'] for r in rows}):
        mature=[r for r in rows if r['history']==history and r['deadline']!='before']
        cancel=[r for r in mature if r['role'] in ('pending_cancel','reissued_cancel')]
        accuracy=lambda records:sum(r['predictions'][r['target_entity']][0]==r['labels'][r['target_entity']][0] for r in records)/len(records)
        primary[str(history)]=accuracy(mature);cancellation[str(history)]=accuracy(cancel)
    ce=F.cross_entropy(torch.tensor([r['logits'] for r in rows]).reshape(-1,2),torch.tensor([r['labels'] for r in rows]).reshape(-1)).item()
    return dict(rows=rows,primary=primary,cancellation=cancellation,cross_entropy=ce,timing_seconds=timing)


def run(arm,output,smoke):
    torch.set_num_threads(1)
    if subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=ROOT,text=True).strip():raise ValueError('freeze source before outcomes')
    source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    protocol=hashlib.sha256((ROOT/'research/dev034/protocol.json').read_bytes()).hexdigest()
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    if (output/(arm+'.json')).exists() or (output/(arm+'.pt')).exists():raise ValueError('preserve outcomes')
    model=build(arm);initial=parameter_hash(model);optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=.002,weight_decay=.0001)
    fps=[];steps=2 if smoke else 216;curve=[];start=time.monotonic();cpu=time.process_time()
    try:
        for step in range(steps):
            cases=training_cases(step,smoke);fps.extend(c.fingerprint for c in cases)
            inputs=public_stream_tensors([c.observations for c in cases]);queries=public_queries([c.tick for c in cases],[c.horizon for c in cases]);targets=labels(cases)
            receipts=[(c.world.act(Action('inspect',random.Random(f'dev034:{step}:{b}').randrange(8))),) for b,c in enumerate(cases)]
            obs=public_stream_tensors(receipts);optimizer.zero_grad(set_to_none=True)
            state=model.encode(*inputs);pre=model.decode(state,queries);post=model.decode(model.update(state,*(x[:,0] for x in obs)),queries)
            loss=(F.cross_entropy(pre.reshape(-1,2),targets.reshape(-1))+F.cross_entropy(post.reshape(-1,2),targets.reshape(-1)))/2
            if not torch.isfinite(loss):raise FloatingPointError('nonfinite loss')
            loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);optimizer.step()
            if step%54==0:
                row=dict(update=step+1,loss=float(loss.detach()),gradient_norm=float(norm));curve.append(row);print(json.dumps(dict(arm=arm,**row)),flush=True)
    except Exception as error:
        torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),rng=torch.get_rng_state(),attempted_update=step+1),output/(arm+'.failure.pt'))
        (output/(arm+'.failure.json')).write_text(json.dumps(dict(source=source,arm=arm,status='failed',attempted_update=step+1,error=str(error),test_accessed=False),indent=2)+'\n')
        raise
    training_seconds=time.monotonic()-start;cpu_seconds=time.process_time()-cpu
    validation=evaluate(model,smoke)
    if set(fps)&{r['fingerprint'] for r in validation['rows']}:raise AssertionError('split overlap')
    checkpoint=output/(arm+'.pt');torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),rng=torch.get_rng_state(),source=source,protocol_sha256=protocol,arm=arm,updates=steps),checkpoint)
    report=dict(experiment='DEV-034',arm=arm,source=source,protocol_sha256=protocol,smoke=smoke,seed=311,initialization_seed=34311,
        updates=steps,learning_rate=.002,initial_hash=initial,final_hash=parameter_hash(model),training_fingerprints=fps,
        checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),parameters=sum(p.numel() for p in model.parameters()),
        trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),state_scalars=model.state_scalars,state_bytes=4*model.state_scalars,
        training_seconds=training_seconds,training_cpu_seconds=cpu_seconds,curve=curve,validation=validation,
        scientific_selection_eligible=False,test_accessed=False,full_system_test6=False,
        runtime=dict(device='cpu',torch_version=str(torch.__version__),threads=torch.get_num_threads()))
    (output/(arm+'.json')).write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(arm=arm,primary=validation['primary'],cancellation=validation['cancellation'],seconds=training_seconds)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--arm',choices=ARMS,required=True);p.add_argument('--output',required=True);p.add_argument('--smoke',action='store_true');a=p.parse_args();run(a.arm,a.output,a.smoke)
