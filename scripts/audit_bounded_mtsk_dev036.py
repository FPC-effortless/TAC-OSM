#!/usr/bin/env python3
"""DEV-036 separate-process checkpoint replay and independent outcome physics."""
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import subprocess
import torch
from run_bounded_mtsk_dev036 import ARMS,TRAINED,build,evaluate,training_cases
from tac_osm.development.delayed_maintenance import Action
from tac_osm.development.noisy_memory import parameter_hash
from tac_osm.development.schema_maintenance import make_pair,execute,ROLES,DEADLINES,exact_public_targets


def audit(folder,allow_incomplete=False):
    torch.set_num_threads(1);folder=Path(folder)
    source=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    protocol=hashlib.sha256(Path('research/dev036/protocol.json').read_bytes()).hexdigest()
    results={};cohort=None;failed=[]
    expected_conditions=set(itertools.product((32,128,512),ROLES,DEADLINES,range(8),(0,1)))
    for arm in ARMS:
        path=folder/(arm+'.json')
        if not path.exists():
            if not allow_incomplete:raise AssertionError(f'missing {arm}')
            failure_path=folder/(arm+'.failure.json')
            if failure_path.exists():
                row=json.loads(failure_path.read_text());snapshot=folder/(arm+'.failure.pt')
                assert row['source']==source and row['protocol_sha256']==protocol
                assert hashlib.sha256(snapshot.read_bytes()).hexdigest()==row['snapshot_sha256']
                failed.append(arm);results[arm]=dict(status='failed',attempted_update=row['attempted_update'],error=row['error'])
            else:
                failed.append(arm);results[arm]=dict(status='missing')
            continue
        r=json.loads(path.read_text());checkpoint=folder/(arm+'.pt')
        assert r['source']==source and r['protocol_sha256']==protocol and not r['smoke']
        assert r['arm']==arm and r['updates']==(216 if arm in TRAINED else 0)
        assert hashlib.sha256(checkpoint.read_bytes()).hexdigest()==r['checkpoint_sha256']
        saved=torch.load(checkpoint,map_location='cpu',weights_only=True)
        assert saved['source']==source and saved['protocol_sha256']==protocol and saved['arm']==arm
        model=build(arm);assert parameter_hash(model)==r['initial_hash']
        model.load_state_dict(saved['model']);assert parameter_hash(model)==r['final_hash']
        assert model.state_scalars==r['state_scalars']
        expected_train=[c.fingerprint for step in range(216) for c in training_cases(step,False)] if arm in TRAINED else []
        assert expected_train==r['training_fingerprints'] and len(set(expected_train))==len(expected_train)
        observed=evaluate(model)
        for field in ('rows','primary','cancellation','all_entity','cross_entropy','physical_totals','selected_minus_none_cost_per_entity'):
            assert observed[field]==r['validation'][field],(arm,field)
        rows=observed['rows']
        assert len(rows)==576 and len({row['fingerprint'] for row in rows})==576
        assert {(x['due_delay'],x['role'],x['deadline'],x['repetition'],x['partner']) for x in rows}==expected_conditions
        assert not set(expected_train)&{row['fingerprint'] for row in rows}
        fingerprints=[x['fingerprint'] for x in rows]
        if cohort is None:cohort=fingerprints
        assert fingerprints==cohort
        for row in rows:
            case=make_pair(910000+row['repetition'],512,row['due_delay'],row['role'],row['deadline'],domain='dev036-v1-holdout')[row['partner']]
            assert row['fingerprint']==case.fingerprint
            assert exact_public_targets(case.observations,case.tick,case.horizon)==case.wait_targets
            assert row['labels']==[[case.wait_targets[e],case.service_targets[e],case.current_targets[e]] for e in range(8)]
            for policy in ('none','selected','matched_random','service_all'):
                actual=row[policy]
                if policy in ('selected','matched_random') and actual['entity'] is not None:
                    receipt=case.world.act(Action('inspect',actual['entity']))
                    assert actual['receipt']==receipt.__dict__
                physics=execute(case.world,case.horizon,tuple(actual['services']),actual['inspections'])
                for k,v in physics.items():assert actual[k]==v,(arm,policy,k)
        results[arm]=dict(status='replayed',primary=observed['primary'],cancellation=observed['cancellation'],
            all_entity=observed['all_entity'],physical_totals=observed['physical_totals'],
            selected_minus_none_cost_per_entity=observed['selected_minus_none_cost_per_entity'],
            checkpoint_sha256=r['checkpoint_sha256'],report_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    return dict(experiment='DEV-036',source=source,protocol_sha256=protocol,complete=not failed,
        replayed_arms=len(ARMS)-len(failed),failed_arms=failed,results=results,scientific_test_accessed=False,
        capability_established=False,limitation='Separate-process exact evaluator replay shares model/evaluation code; independent physics and condition reconstruction, one trained initialization and synthetic known generator')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--output',required=True);p.add_argument('--allow-incomplete',action='store_true')
    a=p.parse_args();result=audit(a.input,a.allow_incomplete);Path(a.output).write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(json.dumps({'complete':result['complete'],'replayed_arms':result['replayed_arms'],'failed_arms':result['failed_arms']}))
