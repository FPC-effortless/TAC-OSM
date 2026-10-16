#!/usr/bin/env python3
"""Check one partial-grid evidence bundle; never select a recipe."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
ROOT=Path(__file__).resolve().parents[1]
PIN='4a8728ef3f08e01d9c542374cf1e722bf8701240'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def summarize(directory):
    directory=Path(directory);manifest=json.loads((directory/'execution-manifest.json').read_text())
    assert manifest['source']==PIN and manifest['learning_rate']==.0005 and manifest['seed']==311
    assert manifest['training_exit_code']==0 and not manifest['timed_out']
    protocol=subprocess.check_output(['git','show',PIN+':research/dev029/protocol.json'],cwd=ROOT)
    summaries=[]
    for budget in (216,648,2160):
        path=directory/'trial'/f'{budget}.json';r=json.loads(path.read_text())
        assert r['source']==PIN and r['protocol_sha256']==hashlib.sha256(protocol).hexdigest()
        assert r['arm']=='gru768' and r['seed']==311 and r['learning_rate']==.0005 and r['curriculum']=='fixed512'
        assert r['updates']==budget and r['split']=='validation' and r['test_accessed'] is False
        assert sha(path.with_suffix('.pt'))==r['checkpoint_sha256']
        replay=directory/f'{budget}-replay.json';rr=json.loads(replay.read_text())
        assert rr['replay_passed'] is True and rr['source']==PIN and rr['updates']==budget
        assert rr['checkpoint_sha256']==r['checkpoint_sha256']
        rows=r['validation']['rows'];assert len(rows)==144
        mature=[row for row in rows if row['deadline']!='before'];assert len(mature)==96
        cancel=[row for row in mature if row['role'] in ('pending_cancel','reissued_cancel')];assert len(cancel)==48
        def accuracy(records):
            return sum(row['predictions'][row['target_entity']][0]==row['labels'][row['target_entity']][0] for row in records)/len(records)
        assert accuracy(mature)==r['primary_accuracy']
        costs={}
        for policy in ('none','selected','matched_random'):
            costs[policy]=dict(failures=sum(row[policy]['failures'] for row in rows),
                services=sum(len(row[policy]['services']) for row in rows),
                unnecessary_services=sum(row[policy]['unnecessary_services'] for row in rows),
                inspections=sum(row[policy]['inspections'] for row in rows),
                total_cost=sum(row[policy]['total_cost'] for row in rows),
                mean_cost_per_entity=sum(row[policy]['cost_per_entity'] for row in rows)/len(rows))
        summaries.append(dict(updates=budget,all_role_validation_accuracy=accuracy(mature),primary_cases=96,
            cancellation_accuracy=accuracy(cancel),cancellation_cases=48,physical_cases=144,physical_costs=costs,
            selected_minus_none_cost=r['validation']['selected_minus_none_cost'],training_seconds=r['training_seconds'],
            parameters=r['parameters'],runtime=r['runtime'],initial_hash=r['initial_hash'],checkpoint_sha256=r['checkpoint_sha256'],
            raw_report_sha256=sha(path),replay_sha256=sha(replay),strict_same_runtime_replay_passed=True))
    return dict(experiment='DEV-032 bounded DEV-029 grid progress',source=PIN,seed=311,learning_rate=.0005,curriculum='fixed512',
        training_complete=True,checkpoints=summaries,full_matrix_complete=False,recipe_selected=False,
        independent_full_matrix_audit_complete=False,test_accessed=False,fresh_seed_confirmation_complete=False,
        original_lr002_trial_remains_failed=True,
        private_raw_evidence='https://www.kaggle.com/code/eweewee2/tac-osm-dev032-registered-gru-slot/versions/1',
        limitation='One initialization, one registered recipe, 96 primary/48 cancellation validation cases; no architecture ranking or convergence claim')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--directory',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    result=summarize(a.directory);Path(a.output).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps([dict(updates=r['updates'],all_role_accuracy=r['all_role_validation_accuracy'],cancellation_accuracy=r['cancellation_accuracy']) for r in result['checkpoints']]))
