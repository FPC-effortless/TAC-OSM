#!/usr/bin/env python3
"""Frozen closed-loop engineering execution, not a trained Test6 capability run."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import torch
from tac_osm.development.integration_test6 import ARMS,MaintenanceSession,PublicRequest,Test6Controller
from tac_osm.development.schema_maintenance import ROLES,make_pair


def run(output):
    if Path(output).exists():raise ValueError('preserve earlier evidence')
    if subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=ROOT,text=True).strip():raise ValueError('freeze implementation first')
    torch.set_num_threads(1);controller=Test6Controller();rows=[];start=time.monotonic()
    for role in ROLES:
        cases=make_pair(32032,32,32,role,'at',domain='dev032-integration-engineering')
        for partner,case in enumerate(cases):
            request=PublicRequest(case.target_entity,case.horizon)
            for arm in ARMS:
                result=controller.rollout(MaintenanceSession(case,request),arm)
                result.update(role=role,partner=partner,public_fingerprint=case.fingerprint)
                rows.append(result)
    fixture_case=make_pair(32033,32,32,'pending_cancel','at',domain='dev032-integration-engineering')[0]
    request=PublicRequest(fixture_case.target_entity,fixture_case.horizon)
    fixtures={name:controller.rollout(MaintenanceSession(fixture_case,request),force_gate=gate)
              for name,gate in (('forced_rejection',0.),('forced_acceptance',1.))}
    assert fixtures['forced_rejection']['repair_executed']
    assert fixtures['forced_rejection']['rejected_commits']==2
    assert not fixtures['forced_acceptance']['repair_executed']
    summary=dict(experiment='DEV-032 Test6 engineering preflight',
        source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        protocol_sha256=hashlib.sha256((ROOT/'research/dev032/protocol.json').read_bytes()).hexdigest(),
        training_performed=False,random_weights=True,full_scientific_test6_complete=False,
        architecture_advantage_established=False,scientific_test_accessed=False,
        original_masked_local_memory_integrated=False,original_boolean_casm_integrated=False,
        exact_cdl_teacher_used=False,parameters=sum(p.numel() for p in controller.parameters()),
        cases=8,arms=list(ARMS),rows=rows,forced_branch_fixtures=fixtures,
        wall_seconds=time.monotonic()-start,
        limitation='New single-entity engineering task; inference ablations at random weights are not trained marginal contributions')
    Path(output).write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps({k:summary[k] for k in ('experiment','source','cases','arms','wall_seconds','full_scientific_test6_complete')}))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();run(a.output)
