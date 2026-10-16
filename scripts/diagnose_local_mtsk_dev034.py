#!/usr/bin/env python3
"""Post-hoc one-batch failure reproduction; no optimizer step or rescued score."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import random
import subprocess
import sys
import torch
PIN='ec0ff80b058308dad8de57eab7d223ec9a81dcba'


def diagnose(checkout,directory,output):
    checkout=Path(checkout);directory=Path(directory);output=Path(output)
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=checkout,text=True).strip()==PIN
    assert not subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=checkout,text=True).strip()
    if output.exists():raise ValueError('preserve earlier diagnostic')
    assert 'tac_osm' not in sys.modules
    spec=importlib.util.spec_from_file_location('frozen_dev034',checkout/'scripts/run_local_mtsk_dev034.py')
    frozen=importlib.util.module_from_spec(spec);spec.loader.exec_module(frozen)
    from tac_osm.development.gradient_diagnostics import gradient_health
    torch.set_num_threads(1);arm='mtsk_local_multi';snapshot=directory/(arm+'.failure.pt')
    saved=torch.load(snapshot,map_location='cpu',weights_only=True)
    failure=json.loads((directory/(arm+'.failure.json')).read_text());assert failure['source']==PIN
    model=frozen.build(arm);model.load_state_dict(saved['model']);torch.set_rng_state(saved['rng'])
    step=saved['attempted_update']-1;cases=frozen.training_cases(step,False)
    inputs=frozen.public_stream_tensors([c.observations for c in cases]);q=frozen.public_queries([c.tick for c in cases],[c.horizon for c in cases]);target=frozen.labels(cases)
    receipts=[(c.world.act(frozen.Action('inspect',random.Random(f'dev034:{step}:{b}').randrange(8))),) for b,c in enumerate(cases)]
    obs=frozen.public_stream_tensors(receipts);state=model.encode(*inputs);before=model.decode(state,q);after=model.decode(model.update(state,*(x[:,0] for x in obs)),q)
    loss=(frozen.F.cross_entropy(before.reshape(-1,2),target.reshape(-1))+frozen.F.cross_entropy(after.reshape(-1,2),target.reshape(-1)))/2;loss.backward()
    report=gradient_health(model.named_parameters());exception=None
    try:torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
    except RuntimeError as error:exception=str(error)
    assert exception is not None,'Original clipping failure did not reproduce'
    report.update(arm=arm,attempted_update=step+1,loss=float(loss.detach()),source=PIN,
        public_case_fingerprints=[c.fingerprint for c in cases],
        failure_snapshot_sha256=hashlib.sha256(snapshot.read_bytes()).hexdigest(),optimizer_steps=0,
        clipping_exception_reproduced=True,original_exception=exception,scientific_trial_remains_failed=True,
        all_preupdate_parameters_finite=all(torch.isfinite(t).all().item() for t in saved['model'].values()),
        scope='Recomputed gradients from frozen pre-update snapshot, not originally captured gradient tensors')
    output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:report[k] for k in ('attempted_update','classification','nonfinite_elements','total_norm_float64')}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--checkout',required=True);p.add_argument('--directory',required=True);p.add_argument('--output',required=True);a=p.parse_args();diagnose(a.checkout,a.directory,a.output)
