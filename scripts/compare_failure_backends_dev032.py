#!/usr/bin/env python3
"""Recompute one frozen failing training batch; no optimizer or test evaluation."""
import argparse
import contextlib
import hashlib
import importlib.util
import json
from pathlib import Path
import random
import subprocess
import sys
import torch
ROOT=Path(__file__).resolve().parents[1]
PIN='4a8728ef3f08e01d9c542374cf1e722bf8701240'


def compare(checkout,evidence,output):
    checkout=Path(checkout);evidence=Path(evidence);output=Path(output)
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=checkout,text=True).strip()==PIN
    assert not subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=checkout,text=True).strip()
    report=json.loads((evidence/'gradient-health.json').read_text());snapshot=evidence/'failure-before-optimizer-step.pt'
    assert hashlib.sha256(snapshot.read_bytes()).hexdigest()==report['failure_snapshot_sha256']
    saved=torch.load(snapshot,map_location='cpu',weights_only=True)
    assert all(torch.isfinite(t).all() for t in saved['model'].values())
    spec=importlib.util.spec_from_file_location('health',ROOT/'src/tac_osm/development/gradient_diagnostics.py')
    health=importlib.util.module_from_spec(spec);spec.loader.exec_module(health)
    assert 'tac_osm' not in sys.modules
    spec=importlib.util.spec_from_file_location('frozen',checkout/'scripts/run_optimization_study_dev029.py')
    frozen=importlib.util.module_from_spec(spec);spec.loader.exec_module(frozen)
    step=report['attempted_update']-1
    cases=frozen.make_pair(311000000+step,512,(32,128,512)[(step//4)%3],frozen.ROLES[step%4],frozen.DEADLINES[(step//12)%3],domain=frozen.domain('train'))
    assert [c.fingerprint for c in cases]==report['public_case_fingerprints']
    inputs=frozen.public_stream_tensors([c.observations for c in cases])
    queries=frozen.public_queries([c.tick for c in cases],[c.horizon for c in cases])
    receipts=[(c.world.act(frozen.Action('inspect',random.Random(f'dev029:{step}:{b}').randrange(8))),) for b,c in enumerate(cases)]
    obs=frozen.public_stream_tensors(receipts)
    output.mkdir(exist_ok=False,parents=True);rows=[];cpu_logits=None
    for name,device,cudnn in [('cpu','cpu',True),('cuda_cudnn','cuda:0',True),('cuda_unfused','cuda:0',False)]:
        model=frozen.DeviceBoundary(frozen.build('gru768',29311),device);model.load_state_dict(saved['model']);model.train()
        if device.startswith('cuda'):assert model.metadata()==report['runtime']
        context=torch.backends.cudnn.flags(enabled=cudnn) if device.startswith('cuda') else contextlib.nullcontext()
        with context:
            state=model.encode(*inputs);before=model.decode(state,queries)
            after=model.decode(model.update(state,*(x[:,0] for x in obs)),queries)
            target=frozen.labels(cases).to(model.device)
            loss=(frozen.F.cross_entropy(before.reshape(-1,2),target.reshape(-1))+frozen.F.cross_entropy(after.reshape(-1,2),target.reshape(-1)))/2
            loss.backward()
        gradients={n:p.grad.detach().cpu() for n,p in model.named_parameters() if p.grad is not None}
        path=output/(name+'-gradients.pt');torch.save(gradients,path)
        logits=torch.stack([before.detach().cpu(),after.detach().cpu()])
        if cpu_logits is None:cpu_logits=logits
        row=health.gradient_health(model.named_parameters())
        row.update(backend=name,runtime=model.metadata(),cudnn_enabled=cudnn,
            loss=float(loss.detach()),max_logits_difference_from_cpu=float((logits-cpu_logits).abs().max()),
            gradients_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        rows.append(row);print(json.dumps({k:row[k] for k in ('backend','classification','nonfinite_elements','loss')}),flush=True)
        del model
    result=dict(frozen_source=PIN,failure_snapshot_sha256=report['failure_snapshot_sha256'],attempted_update=step+1,
        comparison_source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        identical_preupdate_weights=True,identical_public_case_fingerprints=report['public_case_fingerprints'],
        optimizer_steps=0,test_accessed=False,original_trial_remains_failed=True,rows=rows,
        limitation='One-batch backend diagnostic does not establish training convergence or architectural superiority')
    (output/'backend-comparison.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--checkout',required=True);p.add_argument('--evidence',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();compare(a.checkout,a.evidence,a.output)
