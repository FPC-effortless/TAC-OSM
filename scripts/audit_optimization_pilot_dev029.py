#!/usr/bin/env python3
"""Independent checkpoint/physical replay for DEV-029 pilot, not recipe selection."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from tac_osm.development.optimization_program import ARMS
from tac_osm.development.memory_baselines import MemoryBaseline
from tac_osm.development.entity_aging import EntityAgingControl
from tac_osm.development.optimization_program import EventTablePredictor
from tac_osm.development.schema_maintenance import make_pair, execute
from tac_osm.development.delayed_maintenance import Action
from tac_osm.development.noisy_memory import parameter_hash
from run_learned_schema_dev import public_stream_tensors, public_queries


def audit(folder,smoke=False):
    torch.set_num_threads(1)
    folder=Path(folder);reports={};train_bank=None;validation_bank=None;source=None
    for arm in ARMS:
        r=json.loads((folder/f'{arm}.json').read_text())
        assert r['arm']==arm and r['experiment']=='DEV-029'
        assert r['smoke']==smoke and r['updates']==(2 if smoke else 216)
        assert r['scientific_selection_eligible'] is False and r['test_accessed'] is False
        assert r['validation']['split']=='validation'
        assert r['initial_hash']!=r['final_hash']
        protocol_hash=hashlib.sha256((ROOT/'research/dev029/protocol.json').read_bytes()).hexdigest()
        assert r['protocol_sha256']==protocol_hash
        checkpoint=folder/f'{arm}.pt'
        assert hashlib.sha256(checkpoint.read_bytes()).hexdigest()==r['checkpoint_sha256']
        saved=torch.load(checkpoint,map_location='cpu',weights_only=True)
        assert saved['source']==r['source'] and saved['updates']==r['updates']
        assert saved['arm']==arm
        model=(EventTablePredictor(29311) if arm=='event_table' else
               EntityAgingControl('masked_local',29311) if arm=='masked_local' else
               MemoryBaseline('tac_joint' if arm=='native' else arm,29311))
        model.load_state_dict(saved['model']);model.eval()
        assert parameter_hash(model)==r['final_hash']
        expected_train=[c.fingerprint for step in range(r['updates']) for c in make_pair(
            311000000+step,32 if smoke else 512,(32,128,512)[(step//4)%3],
            ('retention','acquisition','pending_cancel','reissued_cancel')[step%4],
            ('before','at','after')[(step//12)%3],domain='dev029-v1-train')]
        assert r['training_fingerprints']==expected_train
        assert len(set(expected_train))==len(expected_train)
        val_rows=r['validation']['rows'];assert len(val_rows)==(8 if smoke else 144)
        import itertools
        expected_conditions=set(itertools.product((32,) if smoke else (32,128,512),
            ('retention','acquisition','pending_cancel','reissued_cancel'),
            ('at',) if smoke else ('before','at','after'),range(1 if smoke else 2),(0,1)))
        assert {(x['due_delay'],x['role'],x['deadline'],x['repetition'],x['partner']) for x in val_rows}==expected_conditions
        observed=[];correct=[];cost_diff=[];physical_totals={'none':{},'selected':{},'matched_random':{}}
        for row in val_rows:
            cases=make_pair(900000+row['repetition'],32 if smoke else 512,row['due_delay'],
                           row['role'],row['deadline'],domain='dev029-v1-validation')
            c=cases[row['partner']];assert row['fingerprint']==c.fingerprint
            observed.append(c.fingerprint)
            assert row['labels']==[[c.wait_targets[e],c.service_targets[e],c.current_targets[e]] for e in range(8)]
            assert row['exact_rule_upper_bound']==list(c.wait_targets)
            with torch.no_grad():
                state=model.encode(*public_stream_tensors([c.observations]))
                q=public_queries([c.tick],[c.horizon]);logits=model.decode(state,q)[0]
            torch.testing.assert_close(logits,torch.tensor(row['logits']),rtol=1e-5,atol=1e-6)
            assert logits.argmax(-1).tolist()==row['predictions']
            probabilities=logits.softmax(-1)[...,1].tolist()
            for kind in physical_totals:
                actual=row[kind]
                if kind=='selected':
                    gains=actual['estimated_gains'];chosen=max(range(8),key=lambda e:(gains[e],-e))
                    chosen=chosen if gains[chosen]>0 else None
                    assert actual['entity']==chosen
                if kind=='matched_random':
                    import random
                    expected_entity=random.Random('dev018-sensing:'+c.fingerprint).randrange(8) if row['selected']['entity'] is not None else None
                    assert actual['entity']==expected_entity
                if kind=='none':
                    p=probabilities;inspections=0
                else:
                    entity=actual['entity'];inspections=int(entity is not None)
                    if entity is None:
                        assert actual['receipt'] is None;p=probabilities
                    else:
                        receipt=c.world.act(Action('inspect',entity))
                        assert actual['receipt']==dict(tick=receipt.tick,entity=entity,kind='inspect',value=receipt.value,due=-1)
                        obs=public_stream_tensors([(receipt,)])
                        with torch.no_grad():
                            p=model.decode(model.update(state,*(x[:,0] for x in obs)),q)[0].softmax(-1)[...,1].tolist()
                    torch.testing.assert_close(torch.tensor(p),torch.tensor(actual['decision_probabilities']),rtol=1e-5,atol=1e-6)
                services=tuple(e for e,values in enumerate(p) if values[0]-values[1]>.25)
                expected=execute(c.world,c.horizon,services,inspections)
                for key,value in expected.items():assert actual[key]==value,(arm,key)
                for key in ('failures','unnecessary_services','inspections','total_cost','cost_per_entity'):
                    physical_totals[kind][key]=physical_totals[kind].get(key,0)+actual[key]
            if row['deadline']!='before':
                correct.append(row['predictions'][c.target_entity][0]==c.wait_targets[c.target_entity])
            cost_diff.append(row['selected']['cost_per_entity']-row['none']['cost_per_entity'])
        assert len(set(observed))==len(observed)
        assert not set(expected_train)&set(observed)
        assert abs(sum(correct)/len(correct)-r['validation']['primary_accuracy'])<1e-12
        assert abs(sum(cost_diff)/len(cost_diff)-r['validation']['selected_minus_none_cost'])<1e-12
        if train_bank is None:train_bank=expected_train;validation_bank=observed;source=r['source']
        assert train_bank==expected_train and validation_bank==observed and source==r['source']
        reports[arm]=dict(validation_accuracy=r['validation']['primary_accuracy'],
            cross_entropy=r['validation']['cross_entropy'],parameters=r['parameters'],
            training_seconds=r['training_seconds'],training_cpu_seconds=r['training_cpu_seconds'],
            state_bytes=r['state_bytes'],physical_totals=physical_totals,
            selected_minus_none_cost=sum(cost_diff)/len(cost_diff))
    return dict(experiment='DEV-029',audit_passed=True,source=source,smoke=smoke,
        arms=reports,paired_validation_episodes=len(validation_bank),
        full_tuning_complete=False,recipe_selected=False,fresh_reliability_complete=False,
        test_accessed=False,architectural_superiority_established=False,
        limitation='Single tuning initialization and one recipe; validation-only engineering pilot; no seed uncertainty estimate')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--output',required=True)
    p.add_argument('--smoke',action='store_true');a=p.parse_args()
    result=audit(a.input,a.smoke);Path(a.output).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))
