#!/usr/bin/env python3
"""Separate-process archive inspection, not a scientific replication."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch


def audit(directory):
    directory=Path(directory)
    report=json.loads((directory/'dynamics-report.json').read_text())
    for filename,key in [('selected-jacobians.pt','jacobian_sha256'),('conditional-gains.pt','conditional_sha256'),('per-event-trace.json','trace_sha256')]:
        assert hashlib.sha256((directory/filename).read_bytes()).hexdigest()==report[key],filename
    matrices=torch.load(directory/'selected-jacobians.pt',weights_only=True)
    gains=torch.load(directory/'conditional-gains.pt',weights_only=True)
    trace=json.loads((directory/'per-event-trace.json').read_text())
    assert len(report['records'])==18
    assert len({(r['snapshot'],r['training_step']) for r in report['records']})==18
    spanning=np.eye(8)[:,:7];spanning[-1]=-1
    basis=np.zeros((96,93));basis[:72,:72]=np.eye(72)
    basis[72:,72:]=np.kron(np.linalg.qr(spanning,mode='reduced')[0],np.eye(3))
    checked=0;failures=[];used=set()
    norm=lambda a:float(np.linalg.svd(a,compute_uv=False)[0])
    for row in report['records']:
        name=row['snapshot'];step=row['training_step'];raw=gains[f'{name}:{step}'].numpy()
        assert np.isfinite(raw).all()
        assert raw.size==row['conditional_row_observations']
        assert int((raw>1.000001).sum())==row['conditional_rows_amplifying']
        assert float(raw.max())==row['maximum_conditional_row_gain']
        entries=[t for t in trace if t['snapshot']==name and t['training_step']==step]
        assert len(entries)==len(raw)*2
        if row['forward_failure'] is None:assert len(raw)==513
        else:failures.append(dict(snapshot=name,step=step,forward_failure=row['forward_failure']))
        expected=set()
        for partner in range(2):
            per_event=raw[:,partner].reshape(len(raw),-1).max(-1)
            for t in [0,255,511,512,int(per_event.argmax())]:
                if t<len(raw):expected.add((partner,t))
            for item in (t for t in entries if t['partner']==partner):
                assert item['conditional_row_gain']==float(per_event[item['position']])
        assert {(p['partner'],p['position']) for p in row['selected_jacobians']}==expected
        for point in row['selected_jacobians']:
            if 'jacobian_failure' in point:
                failures.append(point);continue
            key=point['matrix_key'];used.add(key);matrix=matrices[key].numpy()
            assert matrix.shape==(96,96) and np.isfinite(matrix).all()
            recomputed=dict(ambient_gain=norm(matrix),simplex_tangent_gain=norm(matrix@basis),
                memory_to_memory_gain=norm(matrix[:72,:72]),responsibility_to_memory_gain=norm(matrix[:72,72:]),
                memory_to_responsibility_gain=norm(matrix[72:,:72]),responsibility_to_responsibility_gain=norm(matrix[72:,72:]))
            for k,value in recomputed.items():assert np.isclose(value,point[k],rtol=1e-8,atol=1e-10),(key,k)
            checked+=1
    assert used==set(matrices)
    assert report['optimizer_steps']==0 and not report['scientific_test_accessed']
    result=dict(archive_inspection_passed=True,report_sha256=hashlib.sha256((directory/'dynamics-report.json').read_bytes()).hexdigest(),
        records_checked=18,selected_jacobians_checked=checked,failures_retained=failures,
        scope='Hashes, complete reported census, raw conditional gains and independent NumPy SVDs; not an independent forward rerun or scientific replication')
    (directory/'archive-audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('directory');audit(p.parse_args().directory)
