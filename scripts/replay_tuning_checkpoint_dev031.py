#!/usr/bin/env python3
"""Separate-process same-runtime replay; independent full-matrix audit is pending."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
import torch
from tac_osm.development.compute_runtime import DeviceBoundary
from tac_osm.development.noisy_memory import parameter_hash
from run_optimization_pilot_dev029 import build,validate


def replay(path):
    path=Path(path);report=json.loads(path.read_text());checkpoint=path.with_suffix('.pt')
    assert report['source']==subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    assert report['protocol_sha256']==hashlib.sha256((ROOT/'research/dev029/protocol.json').read_bytes()).hexdigest()
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest()==report['checkpoint_sha256']
    saved=torch.load(checkpoint,map_location='cpu',weights_only=True)
    assert saved['kind']=='DEV-029 tuning' and saved['source']==report['source']
    assert saved['updates']==report['updates'] and saved['runtime']==report['runtime']
    model=DeviceBoundary(build(report['arm'],29000+report['seed']),report['runtime']['device'])
    assert model.metadata()==report['runtime']
    model.load_state_dict(saved['model']);model.eval()
    assert parameter_hash(model)==report['final_hash']
    observed=validate(model,False);expected=report['validation']
    # Same-source/backend replay is strict. Failure is preserved, not rescued
    # by changing prediction, action, cost or floating-point comparison rules.
    for key in ('split','primary_accuracy','cross_entropy','rows','selected_minus_none_cost'):
        if observed[key]!=expected[key]:raise AssertionError('strict replay mismatch: '+key)
    return dict(arm=report['arm'],updates=report['updates'],source=report['source'],
        replay_passed=True,checkpoint_sha256=report['checkpoint_sha256'],
        independent_full_matrix_audit_complete=False,test_accessed=False)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--report',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();result=replay(a.report);Path(a.output).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
