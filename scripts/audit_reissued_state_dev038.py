#!/usr/bin/env python3
"""Separate-process replay of DEV-038 frozen features and probe fits."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import torch
from probe_reissued_state_dev038 import ARM,build,gather,probe_fit
from tac_osm.development.noisy_memory import parameter_hash


def audit(folder,input_dir):
    torch.set_num_threads(1);folder=Path(folder);report=json.loads((folder/'report.json').read_text())
    source=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    assert report['source']==source and report['protocol_sha256']==hashlib.sha256(Path('research/dev038/protocol.json').read_bytes()).hexdigest()
    checkpoint=Path(input_dir)/(ARM+'.pt');assert hashlib.sha256(checkpoint.read_bytes()).hexdigest()==report['checkpoint_sha256']
    model=build(ARM);model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True)['model']);model.eval()
    assert parameter_hash(model)==report['frozen_model_hash']
    records,items=gather(model);assert records==report['records'] and len(records)==192
    assert len({r['fingerprint'] for r in records})==192
    tensor_path=folder/'features-and-probes.pt';assert hashlib.sha256(tensor_path.read_bytes()).hexdigest()==report['tensor_sha256']
    saved=torch.load(tensor_path,map_location='cpu',weights_only=True)
    train=[i for i,r in enumerate(records) if r['rep']<8];test=[i for i,r in enumerate(records) if r['rep']>=8]
    assert len(train)==len(test)==96
    labels=torch.tensor([r['label'] for r in records],dtype=torch.float32)
    for field in ('state','read'):
        x=torch.stack([v[field] for v in items]);assert torch.equal(x,saved[field]['features'])
        probe,mean,scale=probe_fit(x[train],labels[train]);assert torch.equal(mean,saved[field]['mean']) and torch.equal(scale,saved[field]['scale'])
        assert torch.equal(probe.weight,saved[field]['weight']) and torch.equal(probe.bias,saved[field]['bias'])
        with torch.no_grad():probs=probe((x[test]-mean)/scale).sigmoid().flatten()
        assert torch.equal(probs,saved[field]['test_probabilities'])
        predictions=(probs>=.5).int().tolist()
        assert predictions==report['results'][field]['holdout_predictions']
    original=[int(records[i]['original_probability']>=.5) for i in test]
    assert original==report['results']['original']['holdout_predictions']
    result=dict(experiment='DEV-038',source=source,report_sha256=hashlib.sha256((folder/'report.json').read_bytes()).hexdigest(),
        all_192_frozen_features_replayed=True,holdout_labels_reconstructed=True,probe_refit_exact=True,complete=True,
        limitation='Shares frozen predictor and probe code; small synthetic split, one trained model initialization')
    (folder/'audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--checkpoints',required=True)
    a=p.parse_args();audit(a.input,a.checkpoints)
