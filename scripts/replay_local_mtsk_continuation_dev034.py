#!/usr/bin/env python3
"""Separate-process numerical-continuation replay and physical-cost inspection."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import torch
PIN='ec0ff80b058308dad8de57eab7d223ec9a81dcba'


def replay(checkout,directory):
    checkout=Path(checkout);directory=Path(directory)
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=checkout,text=True).strip()==PIN
    assert 'tac_osm' not in sys.modules
    spec=importlib.util.spec_from_file_location('frozen',checkout/'scripts/run_local_mtsk_dev034.py')
    frozen=importlib.util.module_from_spec(spec);spec.loader.exec_module(frozen)
    torch.set_num_threads(1);r=json.loads((directory/'continuation-report.json').read_text())
    assert r['status']=='completed' and r['core_source']==PIN and r['original_trial_remains_failed'] is True
    assert r['scientific_selection_eligible'] is False and r['test_accessed'] is False
    checkpoint=directory/'continuation.pt';assert hashlib.sha256(checkpoint.read_bytes()).hexdigest()==r['checkpoint_sha256']
    saved=torch.load(checkpoint,map_location='cpu',weights_only=True);model=frozen.build('mtsk_local_multi');model.load_state_dict(saved['model'])
    assert saved['core_source']==PIN and saved['intervention_source']==r['intervention_source']
    observed=frozen.evaluate(model,False)
    for field in ('rows','primary','cancellation','cross_entropy'):assert observed[field]==r['validation'][field]
    costs={}
    for row in observed['rows']:
        case=frozen.make_pair(900000+row['repetition'],row['history'],row['due_delay'],row['role'],row['deadline'],domain='dev034-v1-validation')[row['partner']]
        assert case.fingerprint==row['fingerprint']
        for policy in ('none','selected','matched_random'):
            actual=row[policy]
            expected=frozen.execute(case.world,case.horizon,tuple(actual['services']),actual['inspections'])
            for name,value in expected.items():assert actual[name]==value
            bucket=costs.setdefault(str(row['history']),{}).setdefault(policy,dict(failures=0,unnecessary_services=0,inspections=0,total_cost=0.))
            for name in bucket:bucket[name]+=actual[name]
    return dict(scope='Post-hoc numerical diagnostic, not original pilot acceptance',separate_process_exact_replay_passed=True,
        physical_checks_passed=True,core_source=PIN,intervention_source=r['intervention_source'],primary=observed['primary'],
        cancellation=observed['cancellation'],physical_totals=costs,training_seconds=r['training_seconds'],
        continuation_report_sha256=hashlib.sha256((directory/'continuation-report.json').read_bytes()).hexdigest(),
        checkpoint_sha256=r['checkpoint_sha256'],max_gradient_norm_float64=max(p['norm_float64'] for p in r['curve']),
        original_trial_remains_failed=True,architecture_ranking_eligible=False,test_accessed=False)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--checkout',required=True);p.add_argument('--directory',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    result=replay(a.checkout,a.directory);Path(a.output).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
