#!/usr/bin/env python3
"""DEV-029 registered tuning trials, resumable at 216/648/2160 updates.

This runner has no test evaluator. It does not promote engineering-pilot files
into tuning evidence. Each trial starts fresh, or resumes its own source-pinned
checkpoint with optimizer and RNG state. Full tuning has not yet been run.
"""
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
from tac_osm.development.optimization_program import ARMS, LEARNING_RATES, CURRICULA, BUDGETS, domain, history_at
from tac_osm.development.schema_maintenance import make_pair, ROLES, DEADLINES
from tac_osm.development.delayed_maintenance import Action
from tac_osm.development.noisy_memory import parameter_hash
from run_learned_schema_dev import public_stream_tensors, public_queries, labels
from run_optimization_pilot_dev029 import build, validate


def run(args):
    torch.set_num_threads(1)
    source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    if subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=ROOT,text=True).strip():
        raise ValueError('commit the study before executing outcomes')
    protocol=hashlib.sha256((ROOT/'research/dev029/protocol.json').read_bytes()).hexdigest()
    identity=dict(arm=args.arm,learning_rate=args.learning_rate,curriculum=args.curriculum,seed=args.seed)
    model=build(args.arm,29000+args.seed)
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=args.learning_rate,weight_decay=.0001)
    initial=parameter_hash(model);start=0;fps=[];wall_total=cpu_total=0.;events=0
    if args.resume:
        saved=torch.load(args.resume,map_location='cpu',weights_only=True)
        if saved.get('kind')!='DEV-029 tuning' or saved['identity']!=identity or saved['source']!=source or saved['protocol_sha256']!=protocol:
            raise ValueError('resume requires exact registered trial/source; engineering pilot cannot be promoted')
        model.load_state_dict(saved['model']);optimizer.load_state_dict(saved['optimizer'])
        torch.set_rng_state(saved['rng']);start=saved['updates'];fps=saved['training_fingerprints']
        wall_total=saved['training_seconds'];cpu_total=saved['training_cpu_seconds'];events=saved['observed_events']
        initial=saved['initial_hash']
    if not start<args.until:
        raise ValueError('resume checkpoint must precede requested budget')
    output=Path(args.output);output.mkdir(parents=True,exist_ok=True)
    if any((output/f'{b}.json').exists() or (output/f'{b}.pt').exists() for b in BUDGETS if start<b<=args.until):
        raise ValueError('preserve existing outcomes: choose a fresh output directory')
    wall=time.monotonic();cpu=time.process_time()
    for step in range(start,args.until):
        history=history_at(args.curriculum,step)
        cases=make_pair(args.seed*1000000+step,history,(32,128,512)[(step//4)%3],
                        ROLES[step%4],DEADLINES[(step//12)%3],domain=domain('train'))
        fps.extend(c.fingerprint for c in cases);events+=history*2
        inp=public_stream_tensors([c.observations for c in cases])
        q=public_queries([c.tick for c in cases],[c.horizon for c in cases]);target=labels(cases)
        receipts=[(c.world.act(Action('inspect',random.Random(f'dev029:{step}:{b}').randrange(8))),) for b,c in enumerate(cases)]
        obs=public_stream_tensors(receipts)
        model.train();optimizer.zero_grad(set_to_none=True)
        state=model.encode(*inp)
        before=model.decode(state,q);after=model.decode(model.update(state,*(x[:,0] for x in obs)),q)
        loss=(F.cross_entropy(before.reshape(-1,2),target.reshape(-1))+F.cross_entropy(after.reshape(-1,2),target.reshape(-1)))/2
        if not torch.isfinite(loss):raise AssertionError('nonfinite loss')
        loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);optimizer.step()
        if step%108==0:print(json.dumps(dict(identity,update=step,loss=float(loss.detach()))),flush=True)
        if step+1 in BUDGETS:
            wall_total+=time.monotonic()-wall;cpu_total+=time.process_time()-cpu
            model.eval();validation=validate(model,False)
            if set(fps)&{r['fingerprint'] for r in validation['rows']}:raise AssertionError('split overlap')
            checkpoint=output/f'{step+1}.pt'
            torch.save(dict(kind='DEV-029 tuning',identity=identity,source=source,protocol_sha256=protocol,
                updates=step+1,model=model.state_dict(),optimizer=optimizer.state_dict(),rng=torch.get_rng_state(),
                training_fingerprints=fps,initial_hash=initial,training_seconds=wall_total,
                training_cpu_seconds=cpu_total,observed_events=events),checkpoint)
            report=dict(identity,experiment='DEV-029 tuning',source=source,protocol_sha256=protocol,
                updates=step+1,split='validation',smoke=False,initial_hash=initial,final_hash=parameter_hash(model),
                primary_accuracy=validation['primary_accuracy'],cross_entropy=validation['cross_entropy'],
                validation=validation,training_fingerprints=fps.copy(),observed_events=events,
                training_receipts=(step+1)*2,training_seconds=wall_total,training_cpu_seconds=cpu_total,
                parameters=sum(p.numel() for p in model.parameters()),state_bytes=model.state_scalars*4,
                checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),checkpoint_bytes=checkpoint.stat().st_size,
                test_accessed=False,full_matrix_audit_required=True)
            (output/f'{step+1}.json').write_text(json.dumps(report,indent=2)+'\n')
            print(json.dumps(dict(identity,updates=step+1,validation_accuracy=validation['primary_accuracy'])),flush=True)
            wall=time.monotonic();cpu=time.process_time()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--arm',choices=ARMS,required=True)
    p.add_argument('--learning-rate',type=float,choices=LEARNING_RATES,required=True)
    p.add_argument('--curriculum',choices=CURRICULA,required=True)
    p.add_argument('--seed',type=int,choices=(311,419,557),required=True)
    p.add_argument('--until',type=int,choices=BUDGETS,required=True)
    p.add_argument('--resume');p.add_argument('--output',required=True)
    run(p.parse_args())
