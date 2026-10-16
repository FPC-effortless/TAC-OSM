#!/usr/bin/env python3
"""Separate-process exact model replay plus independent physical/census checks."""
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
import torch
from run_local_mtsk_dev034 import ARMS,build,evaluate,training_cases
from tac_osm.development.schema_maintenance import make_pair,execute,ROLES,DEADLINES
from tac_osm.development.delayed_maintenance import Action
from tac_osm.development.noisy_memory import parameter_hash


def audit(folder,smoke=False):
    torch.set_num_threads(1);folder=Path(folder);summaries={};cohort=None
    source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    protocol=hashlib.sha256((ROOT/'research/dev034/protocol.json').read_bytes()).hexdigest()
    for arm in ARMS:
        path=folder/(arm+'.json');r=json.loads(path.read_text());checkpoint=folder/(arm+'.pt')
        assert r['arm']==arm and r['source']==source and r['protocol_sha256']==protocol and r['smoke']==smoke
        assert r['updates']==(2 if smoke else 216) and r['scientific_selection_eligible'] is False and r['test_accessed'] is False
        assert hashlib.sha256(checkpoint.read_bytes()).hexdigest()==r['checkpoint_sha256']
        saved=torch.load(checkpoint,map_location='cpu',weights_only=True)
        assert saved['source']==source and saved['protocol_sha256']==protocol and saved['arm']==arm and saved['updates']==r['updates']
        model=build(arm);assert parameter_hash(model)==r['initial_hash']
        model.load_state_dict(saved['model']);assert parameter_hash(model)==r['final_hash']
        assert model.state_scalars==r['state_scalars']
        expected_train=[c.fingerprint for step in range(r['updates']) for c in training_cases(step,smoke)]
        assert r['training_fingerprints']==expected_train and len(set(expected_train))==len(expected_train)
        observed=evaluate(model,smoke);expected=r['validation']
        for field in ('rows','primary','cancellation','cross_entropy'):assert observed[field]==expected[field],(arm,field)
        rows=expected['rows'];histories=(32,) if smoke else (128,512)
        conditions=set(itertools.product(histories,(32,) if smoke else (32,128,512),ROLES,('at',) if smoke else DEADLINES,range(1 if smoke else 2),(0,1)))
        assert len(rows)==len(conditions)
        assert {(r['history'],r['due_delay'],r['role'],r['deadline'],r['repetition'],r['partner']) for r in rows}==conditions
        fingerprints=[row['fingerprint'] for row in rows]
        assert len(set(fingerprints))==len(fingerprints) and not set(fingerprints)&set(expected_train)
        if cohort is None:cohort=(expected_train,fingerprints)
        assert cohort==(expected_train,fingerprints)
        totals={str(h):{p:dict(failures=0,services=0,unnecessary_services=0,inspections=0,total_cost=0.) for p in ('none','selected','matched_random')} for h in histories}
        for row in rows:
            c=make_pair(900000+row['repetition'],row['history'],row['due_delay'],row['role'],row['deadline'],domain='dev034-v1-validation')[row['partner']]
            assert c.fingerprint==row['fingerprint']
            assert row['labels']==[[c.wait_targets[e],c.service_targets[e],c.current_targets[e]] for e in range(8)]
            for policy in ('none','selected','matched_random'):
                actual=row[policy]
                if policy!='none' and actual['entity'] is not None:
                    receipt=c.world.act(Action('inspect',actual['entity']))
                    assert actual['receipt']==receipt.__dict__
                expected_physics=execute(c.world,c.horizon,tuple(actual['services']),actual['inspections'])
                for name,value in expected_physics.items():assert actual[name]==value,(arm,policy,name)
                bucket=totals[str(row['history'])][policy]
                for name in bucket:bucket[name]+=len(actual[name]) if name=='services' else actual[name]
        costs={}
        for history in histories:
            group=[r for r in rows if r['history']==history]
            costs[str(history)]=sum(r['selected']['cost_per_entity']-r['none']['cost_per_entity'] for r in group)/len(group)
        summaries[arm]=dict(primary=expected['primary'],cancellation=expected['cancellation'],parameters=r['parameters'],
            state_scalars=r['state_scalars'],training_seconds=r['training_seconds'],training_cpu_seconds=r['training_cpu_seconds'],
            inference_timing_seconds=expected['timing_seconds'],physical_totals=totals,selected_minus_none_cost=costs,
            raw_report_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),checkpoint_sha256=r['checkpoint_sha256'])
    interactions={}
    for h in ((32,) if smoke else (128,512)):
        p=lambda arm:summaries[arm]['primary'][str(h)]
        interactions[str(h)]=(p('mtsk_local_multi')-p('mtsk_local_tied'))-(p('mtsk_global_multi')-p('mtsk_global_tied'))
    return dict(experiment='DEV-034',audit_passed=True,source=source,protocol_sha256=protocol,smoke=smoke,
        arms=summaries,descriptive_locality_retention_interaction=interactions,paired_evaluation_cases=len(cohort[1]),
        scientific_test_accessed=False,recipe_selected=False,initialization_reliability_established=False,
        architectural_superiority_established=False,full_system_test6=False,
        limitation='One initialization and limited common recipe on adaptively selected synthetic task; no seed confidence interval or compute equivalence; exact replay shares evaluation code, physical/census checks are independently reconstructed')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--output',required=True);p.add_argument('--smoke',action='store_true');a=p.parse_args()
    result=audit(a.input,a.smoke);Path(a.output).write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(json.dumps(result))
