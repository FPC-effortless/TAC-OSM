#!/usr/bin/env python3
"""Validate returned preflight provenance/completeness and recompute timings.

This CPU-side structural audit does NOT independently rerun CUDA arithmetic.
Numerical parity remains an observed assertion of the frozen CUDA execution.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from tac_osm.development.optimization_program import ARMS
from tac_osm.development.schema_maintenance import make_pair


def audit(path,require_gpu=True):
    raw=Path(path).read_bytes();r=json.loads(raw)
    assert r['experiment']=='DEV-030 compute preflight'
    assert r['scientific_training_run'] is False and r['test_accessed'] is False
    assert set(r['arms'])==set(ARMS)
    protocol=subprocess.check_output(['git','show',r['source']+':research/dev030/protocol.json'],cwd=ROOT)
    assert hashlib.sha256(protocol).hexdigest()==r['protocol_sha256']
    if require_gpu:assert r['gpu_validation_completed'] is True and r['smoke'] is False
    expected_samples=1 if r['smoke'] else 3
    rows={};cpu_total=gpu_total=0.
    for arm,row in r['arms'].items():
        for backend in ('cpu','cuda') if require_gpu else ('cpu',):
            measurement=row[backend];samples=measurement['seconds_per_update']
            assert len(samples)==expected_samples and all(math.isfinite(x) and x>0 for x in samples)
            assert statistics.median(samples)==measurement['median_seconds_per_update']
            runtime=measurement['runtime']
            assert runtime['dtype']=='float32' and not runtime['tf32'] and not runtime['amp'] and not runtime['compilation']
            assert runtime['batch_size_unchanged'] is True
            assert runtime['device']==('cpu' if backend=='cpu' else 'cuda:0')
            assert runtime['state_device']==('cpu' if backend=='cpu' or arm=='event_table' else 'cuda:0')
        cpu=row['cpu']['median_seconds_per_update'];cpu_total+=cpu
        rows[arm]=dict(cpu_seconds_per_update=cpu)
        if require_gpu:
            gpu=row['cuda']['median_seconds_per_update'];gpu_total+=gpu
            runtime=row['cuda']['runtime'];assert runtime['gpu_name'] and runtime['cuda_version']
            assert runtime['torch_version']==row['cpu']['runtime']['torch_version']
            p=row['parity'];assert p['passed'] and p['initial_categorical_predictions_exact'] and p['initial_service_choices_exact']
            assert all(math.isfinite(p[k]) and p[k]>=0 for k in ('initial_max_abs_error','post_update_max_abs_error'))
            pair=make_pair(303002,512,32,'pending_cancel','at',domain='dev030-engineering-fixture')
            assert p['fixture_fingerprints']==[c.fingerprint for c in pair]
            rows[arm].update(cuda_seconds_per_update=gpu,cpu_over_cuda_speedup=cpu/gpu,
                gpu_name=runtime['gpu_name'],torch_version=runtime['torch_version'],
                initial_max_abs_error=p['initial_max_abs_error'],post_update_max_abs_error=p['post_update_max_abs_error'])
    recommended=require_gpu and gpu_total<=cpu_total*.8
    assert r['recommendation']['prefer_gpu']==recommended
    if require_gpu:
        assert abs(r['recommendation']['aggregate_cpu_over_gpu_speedup']-cpu_total/gpu_total)<1e-12
    return dict(experiment='DEV-030 returned evidence structural audit',structural_audit_passed=True,
        source=r['source'],raw_sha256=hashlib.sha256(raw).hexdigest(),gpu_evidence_present=require_gpu,
        recommend_gpu=recommended,arms=rows,
        aggregate_cpu_over_cuda_speedup=cpu_total/gpu_total if require_gpu else None,
        scientific_test_accessed=False,full_tuning_complete=False,
        limitation='Recomputes provenance, coverage and timing recommendation; CUDA parity assertions are from the frozen execution, not an independent CUDA arithmetic replay')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--output',required=True)
    p.add_argument('--cpu-only',action='store_true');a=p.parse_args();result=audit(a.input,not a.cpu_only)
    Path(a.output).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
