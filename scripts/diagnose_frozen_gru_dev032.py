#!/usr/bin/env python3
"""Instrument the original frozen runner; do not rescue or reinterpret failure.

Requires a clean isolated 4a8728e checkout, its saved216 GRU checkpoint/report,
and exactly its recorded runtime. No exposed credential is accepted or used.
"""
import argparse
import hashlib
import importlib.util
import inspect
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import torch
ROOT=Path(__file__).resolve().parents[1]
PIN='4a8728ef3f08e01d9c542374cf1e722bf8701240'


def diagnose(checkout,report_path,output):
    checkout=Path(checkout).resolve();report_path=Path(report_path).resolve();output=Path(output).resolve()
    if subprocess.check_output(['git','rev-parse','HEAD'],cwd=checkout,text=True).strip()!=PIN:raise ValueError('original source required')
    if subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=checkout,text=True).strip():raise ValueError('original checkout must be clean')
    report=json.loads(report_path.read_text());checkpoint=report_path.with_suffix('.pt')
    if report['source']!=PIN or report['arm']!='gru768' or report['updates']!=216:raise ValueError('requires original GRU216 report')
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest()!=report['checkpoint_sha256']:raise ValueError('checkpoint hash mismatch')
    if output.exists():raise ValueError('preserve old failure evidence; choose fresh output')
    output.mkdir(parents=True)
    # Load diagnostics by file, so the model package can resolve exclusively
    # from the original checkout when its frozen runner is imported.
    health_spec=importlib.util.spec_from_file_location('gradient_health_tool',ROOT/'src/tac_osm/development/gradient_diagnostics.py')
    health=importlib.util.module_from_spec(health_spec);health_spec.loader.exec_module(health)
    if 'tac_osm' in sys.modules:raise ValueError('run in a fresh process; model package already imported')
    frozen_spec=importlib.util.spec_from_file_location('frozen_dev031_runner',checkout/'scripts/run_optimization_study_dev029.py')
    frozen=importlib.util.module_from_spec(frozen_spec);frozen_spec.loader.exec_module(frozen)
    original=torch.nn.utils.clip_grad_norm_;captured=[]
    def observe(parameters,*args,**kwargs):
        parameters=list(parameters)
        try:return original(parameters,*args,**kwargs)
        except RuntimeError as error:
            frame=inspect.currentframe().f_back
            model=frame.f_locals['model'];optimizer=frame.f_locals['optimizer'];step=frame.f_locals['step']
            diagnostic=health.gradient_health(model.named_parameters())
            diagnostic.update(frozen_source=PIN,diagnostic_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                attempted_update=step+1,runtime=model.metadata(),original_exception=str(error),
                finite_forward_loss=float(frame.f_locals['loss'].detach()),
                public_case_fingerprints=[c.fingerprint for c in frame.f_locals['cases']],
                original_error_if_nonfinite=kwargs.get('error_if_nonfinite'),scientific_trial_remains_failed=True)
            snapshot=output/'failure-before-optimizer-step.pt'
            torch.save(dict(model={n:t.detach().cpu() for n,t in model.state_dict().items()},
                gradients={n:p.grad.detach().cpu() for n,p in model.named_parameters() if p.grad is not None},
                optimizer=optimizer.state_dict(),cpu_rng=torch.get_rng_state(),
                cuda_rng=torch.cuda.get_rng_state_all() if model.device.type=='cuda' else []),snapshot)
            diagnostic['failure_snapshot_sha256']=hashlib.sha256(snapshot.read_bytes()).hexdigest()
            (output/'gradient-health.json').write_text(json.dumps(diagnostic,indent=2,allow_nan=False)+'\n')
            captured.append(diagnostic);raise
    torch.nn.utils.clip_grad_norm_=observe
    try:
        frozen.run(SimpleNamespace(arm='gru768',seed=311,learning_rate=.002,curriculum='fixed512',until=648,
            device=report['runtime']['device'],resume=str(checkpoint),output=str(output/'reproduction')))
    finally:
        torch.nn.utils.clip_grad_norm_=original
        (output/'diagnostic-status.json').write_text(json.dumps(dict(failure_captured=bool(captured),
            original_failure_not_rescued=True,gpu_reproduction_claimed=bool(captured),test_accessed=False),indent=2)+'\n')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--checkout',required=True);p.add_argument('--report',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();diagnose(a.checkout,a.report,a.output)
