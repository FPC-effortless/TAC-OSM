#!/usr/bin/env python3
"""Independently check retained gradient tensors on CPU, without recomputation."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import torch


def audit(directory):
    directory=Path(directory);path=directory/'backend-comparison.json';r=json.loads(path.read_text())
    assert r['optimizer_steps']==0 and r['test_accessed'] is False and r['original_trial_remains_failed'] is True
    assert r['identical_preupdate_weights'] is True
    assert r['frozen_source']=='4a8728ef3f08e01d9c542374cf1e722bf8701240'
    assert {row['backend'] for row in r['rows']}=={'cpu','cuda_cudnn','cuda_unfused'} and len(r['rows'])==3
    summaries=[]
    for row in r['rows']:
        file=directory/(row['backend']+'-gradients.pt')
        assert hashlib.sha256(file.read_bytes()).hexdigest()==row['gradients_sha256']
        gradients=torch.load(file,map_location='cpu',weights_only=True)
        reported={p['name']:p for p in row['parameters']};assert set(gradients)==set(reported)
        bad=0
        for name,tensor in gradients.items():
            count=int((~torch.isfinite(tensor)).sum());bad+=count
            assert reported[name]['elements']==tensor.numel() and reported[name]['nonfinite_elements']==count
        assert bad==row['nonfinite_elements']
        assert math.isfinite(row['loss'])
        if bad:assert row['classification']=='nonfinite_gradient_elements'
        summaries.append(dict(backend=row['backend'],nonfinite_elements=bad,gradient_sha256=row['gradients_sha256']))
    return dict(status='passed',report_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        rows=summaries,all_backends_have_nonfinite_gradients=all(row['nonfinite_elements'] for row in summaries),
        limitation='Independent inspection of retained tensors, not an independent forward/backward rerun')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--directory',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    result=audit(a.directory);Path(a.output).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
