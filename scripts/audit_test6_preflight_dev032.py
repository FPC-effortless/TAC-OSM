#!/usr/bin/env python3
"""Replay physical actions independently of the policy/session implementation."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from tac_osm.development.delayed_maintenance import Action
from tac_osm.development.schema_maintenance import ROLES,make_pair,future_bits

ARMS={'full','memory_reset','cdl_uniform','casm_mlp','osm_off','commit_all','repair_off'}


def check_row(row,case):
    world=copy.deepcopy(case.world);services=inspections=0
    untreated=future_bits(case.world,case.horizon)[case.target_entity]
    trace=row['trace']
    assert 1<=len(trace)<=2
    assert row['repair_executed']==(len(trace)==2)
    assert row['rejected_commits']==sum(not t['commit_accepted'] for t in trace)
    assert row['accepted_commits']==sum(t['commit_accepted'] for t in trace)
    for attempt,t in enumerate(trace):
        assert t['attempt']==attempt
        action=t['action'];assert action in (0,1,2)
        if attempt:assert not trace[0]['commit_accepted']
        if action==0:command=Action('wait',case.target_entity)
        elif action==1:
            command=Action('service',case.target_entity,0,case.horizon);services+=1
        else:command=Action('inspect',case.target_entity);inspections+=1
        assert world.act(command).__dict__==t['public_receipt']
        assert t['receipt_acceptance'] is True
    failures=future_bits(world,case.horizon)[case.target_entity]
    expected=dict(target_failures=failures,services=services,
        unnecessary_services=services*int(untreated==0),inspections=inspections,
        total_cost=4*failures+services+.1*inspections,
        failure_scope='queried public entity only',all_entity_benchmark=False)
    assert row['cost']==expected
    assert row['capability_validated'] is False
    assert row['future_targets_supplied_to_policy'] is False
    if row['arm']=='repair_off':assert len(trace)==1
    return len(trace)


def audit(path):
    report=json.loads(Path(path).read_text())
    protocol=subprocess.check_output(['git','show',report['source']+':research/dev032/protocol.json'],cwd=ROOT)
    assert hashlib.sha256(protocol).hexdigest()==report['protocol_sha256']
    for field in ('training_performed','full_scientific_test6_complete','architecture_advantage_established','scientific_test_accessed'):
        assert report[field] is False
    assert report['random_weights'] is True and report['cases']==8
    assert set(report['arms'])==ARMS and len(report['rows'])==56
    seen=set();actions=0
    for row in report['rows']:
        key=(row['role'],row['partner'],row['arm']);assert key not in seen;seen.add(key)
        case=make_pair(32032,32,32,row['role'],'at',domain='dev032-integration-engineering')[row['partner']]
        assert row['public_fingerprint']==case.fingerprint
        actions+=check_row(row,case)
    assert seen=={(role,partner,arm) for role in ROLES for partner in (0,1) for arm in ARMS}
    case=make_pair(32033,32,32,'pending_cancel','at',domain='dev032-integration-engineering')[0]
    assert set(report['forced_branch_fixtures'])=={'forced_rejection','forced_acceptance'}
    for name,row in report['forced_branch_fixtures'].items():
        actions+=check_row(row,case);assert row['forced_gate_fixture'] is True
        assert row['repair_executed']==(name=='forced_rejection')
    return dict(status='passed',source=report['source'],report_sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest(),
        ordinary_rollouts=56,branch_fixtures=2,replayed_actions=actions,
        scope='Independent physical receipt/cost replay; not a trained capability or independent rerun of neural predictions')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--report',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    result=audit(a.report);Path(a.output).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
