#!/usr/bin/env python3
"""DEV-039 separate-process frozen-feature, head-fit and physical replay."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import torch
from run_predictive_head_dev039 import ARM,build,encoded,evaluate,fit
from tac_osm.development.noisy_memory import parameter_hash
from tac_osm.development.schema_maintenance import execute


def audit(folder,input_dir):
    torch.set_num_threads(1);folder=Path(folder);r=json.loads((folder/'report.json').read_text())
    source=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    assert source==r['source'] and r['protocol_sha256']==hashlib.sha256(Path('research/dev039/protocol.json').read_bytes()).hexdigest()
    checkpoint=Path(input_dir)/(ARM+'.pt');assert hashlib.sha256(checkpoint.read_bytes()).hexdigest()==r['checkpoint_sha256']
    model=build(ARM);model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True)['model']);model.eval()
    assert parameter_hash(model)==r['model_hash']
    raw_path=folder/'features-heads.pt';assert hashlib.sha256(raw_path.read_bytes()).hexdigest()==r['raw_sha256']
    raw=torch.load(raw_path,map_location='cpu',weights_only=True)
    train_cases,train_x,train_y,train_c,_=encoded(model,'dev039-v1-train')
    eval_cases,eval_x,eval_y,eval_c,_=encoded(model,'dev039-v1-eval')
    assert len(train_cases)==len(eval_cases)==576
    assert train_c==r['train_conditions'] and eval_c==r['eval_conditions']
    assert len({c.fingerprint for c in train_cases})==576 and len({c.fingerprint for c in eval_cases})==576
    assert not {c.fingerprint for c in train_cases}&{c.fingerprint for c in eval_cases}
    for key,value in [('train_features',train_x),('train_labels',train_y),('eval_features',eval_x),('eval_labels',eval_y)]:
        assert torch.equal(raw[key],value)
    assert all(torch.equal(raw['original'][k],v) for k,v in model.head.state_dict().items())
    heads={'original':model.head,'refit_mlp':torch.nn.Sequential(torch.nn.Linear(44,64),torch.nn.Tanh(),torch.nn.Linear(64,2))}
    heads['refit_mlp'].load_state_dict(model.head.state_dict())
    torch.manual_seed(39039);heads['linear']=torch.nn.Linear(44,2)
    fit(heads['refit_mlp'],train_x,train_y,.002);fit(heads['linear'],train_x,train_y,.02)
    for name,head in heads.items():
        assert all(torch.equal(raw[name][k],v) for k,v in head.state_dict().items())
        observed=evaluate(head,eval_x,eval_y,eval_cases,eval_c);saved=r['results'][name]
        for field in ('rows','primary','cancellation','reissued','all_entity','physical_totals','cross_entropy'):
            assert observed[field]==saved[field],(name,field)
        for row,case in zip(saved['rows'],eval_cases):
            physics=execute(case.world,case.horizon,tuple(row['services']),0)
            for k,v in physics.items():assert row['physical'][k]==v,(name,k)
    result=dict(experiment='DEV-039',source=source,report_sha256=hashlib.sha256((folder/'report.json').read_bytes()).hexdigest(),
        worlds_per_namespace=576,checkpoint_and_features_replayed=True,both_head_fits_exact=True,
        independent_physical_reconstruction_passed=True,complete=True,
        limitation='Same frozen model/head code; one initialization and synthetic generator, not independent capability confirmation')
    (folder/'audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--checkpoints',required=True)
    a=p.parse_args();audit(a.input,a.checkpoints)
