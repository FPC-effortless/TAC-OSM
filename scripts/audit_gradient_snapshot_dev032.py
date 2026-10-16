#!/usr/bin/env python3
"""Independent CPU inspection of captured tensors, without model execution."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import torch


def audit(directory):
    directory=Path(directory);r=json.loads((directory/'gradient-health.json').read_text())
    snapshot=directory/'failure-before-optimizer-step.pt'
    assert hashlib.sha256(snapshot.read_bytes()).hexdigest()==r['failure_snapshot_sha256']
    s=torch.load(snapshot,map_location='cpu',weights_only=True)
    rows={row['name']:row for row in r['parameters']}
    assert set(rows)==set(s['gradients'])
    for name,t in s['gradients'].items():
        assert rows[name]['elements']==t.numel()
        assert rows[name]['nonfinite_elements']==int((~torch.isfinite(t)).sum())
    nans=sum(int(torch.isnan(t).sum()) for t in s['gradients'].values())
    infinities=sum(int(torch.isinf(t).sum()) for t in s['gradients'].values())
    assert nans+infinities==r['nonfinite_elements']
    if nans+infinities:assert r['classification']=='nonfinite_gradient_elements'
    assert math.isfinite(r['finite_forward_loss'])
    assert r['original_error_if_nonfinite'] is True and r['scientific_trial_remains_failed'] is True
    return dict(status='passed',snapshot_sha256=r['failure_snapshot_sha256'],nonfinite_elements_verified=nans+infinities,
        all_preupdate_parameters_finite=all(torch.isfinite(t).all().item() for t in s['model'].values()),
        frozen_source=r['frozen_source'],attempted_update=r['attempted_update'],finite_forward_loss=r['finite_forward_loss'],
        original_exception_preserved=True,optimizer_step_after_failure=False,
        scope='Independent CPU inspection of saved gradient tensors, not independent retraining',
        private_remote_evidence='https://www.kaggle.com/code/eweewee2/tac-osm-dev032-frozen-gru-diagnosis/versions/2',
        nan_gradient_elements=nans,infinite_gradient_elements=infinities)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--directory',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    result=audit(a.directory);Path(a.output).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
