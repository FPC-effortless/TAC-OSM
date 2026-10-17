#!/usr/bin/env python3
"""DEV-037 separate-process replay with reconstructed simulator action costs."""
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import subprocess
import torch
from run_simulator_action_dev037 import ARM,build,choose,choose_threshold,gather,policy,teacher_action
from tac_osm.development.noisy_memory import parameter_hash
from tac_osm.development.schema_maintenance import execute,ROLES,DEADLINES


def audit(folder,input_dir):
    torch.set_num_threads(1);folder=Path(folder)
    row=json.loads((folder/'report.json').read_text())
    source=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    assert row['source']==source and row['protocol_sha256']==hashlib.sha256(Path('research/dev037/protocol.json').read_bytes()).hexdigest()
    checkpoint=Path(input_dir)/(ARM+'.pt');assert hashlib.sha256(checkpoint.read_bytes()).hexdigest()==row['frozen_checkpoint_sha256']
    model=build(ARM);model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True)['model']);model.eval()
    assert parameter_hash(model)==row['frozen_model_hash']
    policies=folder/'policies.pt';assert hashlib.sha256(policies.read_bytes()).hexdigest()==row['policies_sha256']
    saved=torch.load(policies,map_location='cpu',weights_only=True)
    assert set(saved)=={'active','random'}
    pool_cases,pool,_=gather(model,'dev037-v1-pool')
    assert pool==row['pool_public'] and len(pool)==576
    selected={name:choose(pool,name) for name in ('random','active')}
    for name,indices in selected.items():
        assert indices==row['training'][name]['selected_indices']
        assert [pool[i]['fingerprint'] for i in indices]==row['training'][name]['selected_fingerprints']
    threshold,_=choose_threshold(pool,pool_cases,selected['random'])
    assert threshold==row['tuned_threshold']
    models={name:policy() for name in saved}
    for name in saved:models[name].load_state_dict(saved[name]);models[name].eval()
    eval_cases,public,_=gather(model,'dev037-v1-eval')
    assert len(public)==576 and len({p['fingerprint'] for p in public})==576
    assert not {p['fingerprint'] for p in public}&{p['fingerprint'] for p in pool}
    expected=set(itertools.product((32,128,512),ROLES,DEADLINES,range(8),(0,1)))
    assert {(x['delay'],x['role'],x['deadline'],x['rep'],x['partner']) for x in public}==expected
    totals={name:dict(failures=0,unnecessary_services=0,inspections=0,total_cost=0.,services=0) for name in row['evaluation_totals']}
    for p,case,saved_row in zip(public,eval_cases,row['evaluation_rows']):
        assert saved_row['fingerprint']==case.fingerprint and saved_row['probabilities']==p['probabilities']
        assert saved_row['teacher_actions']==[e for e in range(8) if teacher_action(case.wait_targets[e],case.service_targets[e])]
        features=torch.tensor(p['probabilities'],dtype=torch.float32)
        with torch.no_grad():
            decisions={
                'fixed_threshold':tuple(e for e,v in enumerate(p['probabilities']) if v[0]-v[1]>.25),
                'tuned_threshold':tuple(e for e,v in enumerate(p['probabilities']) if v[0]-v[1]>threshold),
                'random_teacher':tuple(e for e,v in enumerate(models['random'](features).squeeze(-1)) if float(v)>=0),
                'active_teacher':tuple(e for e,v in enumerate(models['active'](features).squeeze(-1)) if float(v)>=0),
                'oracle':tuple(saved_row['teacher_actions'])}
        for name,services in decisions.items():
            actual=saved_row['outcomes'][name]
            assert tuple(actual['services'])==services
            independent=execute(case.world,case.horizon,services,0)
            for k,v in independent.items():assert actual[k]==v,(name,k)
            for k in ('failures','unnecessary_services','inspections','total_cost'):totals[name][k]+=actual[k]
            totals[name]['services']+=len(services)
    for name,subtotal in totals.items():
        for k,v in subtotal.items():assert v==row['evaluation_totals'][name][k]
    result=dict(experiment='DEV-037',source=source,report_sha256=hashlib.sha256((folder/'report.json').read_bytes()).hexdigest(),
        checkpoint_hash_passed=True,model_prediction_replay_passed=True,selection_census_passed=True,
        independently_reconstructed_physics_passed=True,worlds=576,complete=True,
        limitation='Shared frozen model inference and teacher rule code; one synthetic generator and adaptive checkpoint selection, not independent scientific confirmation')
    (folder/'audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--checkpoints',required=True)
    a=p.parse_args();audit(a.input,a.checkpoints)
