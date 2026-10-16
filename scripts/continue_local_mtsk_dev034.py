#!/usr/bin/env python3
"""Registered post-hoc numerical continuation, never a replacement pilot cell."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import random
import subprocess
import sys
import time
import torch
ROOT=Path(__file__).resolve().parents[1]
PIN='ec0ff80b058308dad8de57eab7d223ec9a81dcba'


def run(checkout,directory,output):
    checkout=Path(checkout);directory=Path(directory);output=Path(output)
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=checkout,text=True).strip()==PIN
    assert not subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=ROOT,text=True).strip()
    if output.exists():raise ValueError('preserve original and continuation outcomes')
    output.mkdir(parents=True)
    spec=importlib.util.spec_from_file_location('candidate_clipping',ROOT/'src/tac_osm/development/finite_clipping.py')
    clipping=importlib.util.module_from_spec(spec);spec.loader.exec_module(clipping)
    assert 'tac_osm' not in sys.modules
    spec=importlib.util.spec_from_file_location('frozen_dev034',checkout/'scripts/run_local_mtsk_dev034.py')
    frozen=importlib.util.module_from_spec(spec);spec.loader.exec_module(frozen)
    torch.set_num_threads(1);arm='mtsk_local_multi';snapshot=directory/(arm+'.failure.pt')
    saved=torch.load(snapshot,map_location='cpu',weights_only=True);model=frozen.build(arm);model.load_state_dict(saved['model'])
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=.002,weight_decay=.0001)
    optimizer.load_state_dict(saved['optimizer']);torch.set_rng_state(saved['rng'])
    source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();curve=[];start=time.monotonic();start_step=saved['attempted_update']-1
    report=dict(experiment='DEV-034 post-hoc numerical continuation',core_source=PIN,intervention_source=source,
        failure_snapshot_sha256=hashlib.sha256(snapshot.read_bytes()).hexdigest(),start_completed_updates=start_step,
        planned_final_updates=216,original_trial_remains_failed=True,scientific_selection_eligible=False,
        architecture_ranking_eligible=False,test_accessed=False,clipping='FP64 norm, finite CPU FP32 elements only')
    try:
        for step in range(start_step,216):
            cases=frozen.training_cases(step,False);inputs=frozen.public_stream_tensors([c.observations for c in cases]);q=frozen.public_queries([c.tick for c in cases],[c.horizon for c in cases]);target=frozen.labels(cases)
            receipts=[(c.world.act(frozen.Action('inspect',random.Random(f'dev034:{step}:{b}').randrange(8))),) for b,c in enumerate(cases)];obs=frozen.public_stream_tensors(receipts)
            optimizer.zero_grad(set_to_none=True);state=model.encode(*inputs);before=model.decode(state,q);after=model.decode(model.update(state,*(x[:,0] for x in obs)),q)
            loss=(frozen.F.cross_entropy(before.reshape(-1,2),target.reshape(-1))+frozen.F.cross_entropy(after.reshape(-1,2),target.reshape(-1)))/2
            if not torch.isfinite(loss):raise FloatingPointError('nonfinite loss')
            loss.backward();norm=clipping.clip_finite_cpu_gradients_(model.parameters());optimizer.step()
            curve.append(dict(update=step+1,loss=float(loss.detach()),norm_float64=norm))
            if (step-start_step)%20==0:print(json.dumps(curve[-1]),flush=True)
        report.update(status='completed',training_seconds=time.monotonic()-start,validation=frozen.evaluate(model,False))
        # Independent model object, same frozen evaluator: exact prediction and
        # full action/cost replay, not an independent scientific replication.
        checkpoint=output/'continuation.pt'
        torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),rng=torch.get_rng_state(),core_source=PIN,intervention_source=source),checkpoint)
        replay=frozen.build(arm);replay.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True)['model'])
        observed=frozen.evaluate(replay,False)
        for field in ('rows','primary','cancellation','cross_entropy'):assert observed[field]==report['validation'][field]
        report.update(checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),exact_reload_replay_passed=True)
    except Exception as error:
        report.update(status='failed',attempted_update=step+1,error=str(error))
        torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),rng=torch.get_rng_state()),output/'continuation-failure.pt')
        raise
    finally:
        report.update(curve=curve,wall_seconds=time.monotonic()-start)
        (output/'continuation-report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
        print(json.dumps({k:report[k] for k in ('status','original_trial_remains_failed','wall_seconds')}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--checkout',required=True);p.add_argument('--directory',required=True);p.add_argument('--output',required=True);a=p.parse_args();run(a.checkout,a.directory,a.output)
