#!/usr/bin/env python3
"""Frozen local MTSK state versus trained reader on new cancellation pairs."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
import torch
import torch.nn.functional as F
from run_bounded_mtsk_dev036 import build
from run_learned_schema_dev import public_stream_tensors,public_queries
from tac_osm.development.schema_maintenance import make_pair
from tac_osm.development.noisy_memory import parameter_hash
ARM='bounded_local_multi_fp64'


def probe_fit(train_x,train_y):
    torch.manual_seed(38038)
    mean=train_x.mean(0);scale=train_x.std(0,unbiased=False).clamp_min(1e-3)
    x=(train_x-mean)/scale
    probe=torch.nn.Linear(x.shape[-1],1)
    opt=torch.optim.Adam(probe.parameters(),lr=.03)
    for _ in range(500):
        opt.zero_grad(set_to_none=True)
        loss=F.binary_cross_entropy_with_logits(probe(x).flatten(),train_y)+.01*probe.weight.square().sum()
        loss.backward();opt.step()
    return probe,mean,scale


@torch.no_grad()
def gather(model):
    records=[];features=[]
    for delay in (32,128,512):
        for deadline in ('at','after'):
            for rep in range(16):
                cases=make_pair(920000+rep,512,delay,'reissued_cancel',deadline,domain='dev038-v1-probe')
                inputs=public_stream_tensors([c.observations for c in cases]);queries=public_queries([c.tick for c in cases],[c.horizon for c in cases])
                state=model.encode(*inputs);q=model.query_encoder(queries);read=model.memory.read(state,q);logits=model.decode(state,queries)
                probs=logits.softmax(-1)[...,1]
                for b,c in enumerate(cases):
                    e=c.target_entity;local=state[b,e];query=q[b,e,0];read_wait=read[b,e,0]
                    features.append(dict(state=torch.cat((local,query)).detach().clone(),read=torch.cat((read_wait,query)).detach().clone(),
                        original_probability=float(probs[b,e,0]),label=float(c.wait_targets[e])))
                    records.append(dict(delay=delay,deadline=deadline,rep=rep,partner=b,fingerprint=c.fingerprint,target_entity=e,
                        label=int(c.wait_targets[e]),original_probability=float(probs[b,e,0])))
    return records,features


def run(input_dir,output):
    torch.set_num_threads(1);output=Path(output)
    if output.exists():raise ValueError('preserve prior probe')
    if subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=ROOT,text=True).strip():raise ValueError('freeze source before probe')
    source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    protocol=hashlib.sha256((ROOT/'research/dev038/protocol.json').read_bytes()).hexdigest()
    prior=json.loads((ROOT/'docs/DEV-036-AUDITED-SUMMARY.json').read_text())['arms'][ARM]['checkpoint_sha256']
    checkpoint=Path(input_dir)/(ARM+'.pt');assert hashlib.sha256(checkpoint.read_bytes()).hexdigest()==prior
    model=build(ARM);model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True)['model']);model.eval()
    records,items=gather(model)
    assert len(records)==192 and len({r['fingerprint'] for r in records})==192
    train=[i for i,r in enumerate(records) if r['rep']<8];test=[i for i,r in enumerate(records) if r['rep']>=8]
    assert len(train)==len(test)==96
    tensors={};results={};targets=torch.tensor([x['label'] for x in records],dtype=torch.float32)
    for field in ('state','read'):
        values=torch.stack([x[field] for x in items]);probe,mean,scale=probe_fit(values[train],targets[train])
        with torch.no_grad():estimated=probe((values[test]-mean)/scale).sigmoid().flatten()
        predictions=(estimated>=.5).int().tolist();truth=targets[test].int().tolist()
        tensors[field]=dict(features=values,weight=probe.weight.detach(),bias=probe.bias.detach(),mean=mean,scale=scale,
            test_probabilities=estimated.detach())
        results[field]=dict(train_worlds=len(train),holdout_worlds=len(test),holdout_accuracy=sum(a==b for a,b in zip(predictions,truth))/len(test),
            holdout_predictions=predictions,holdout_probabilities=estimated.tolist())
    original=[int(records[i]['original_probability']>=.5) for i in test]
    results['original']=dict(holdout_accuracy=sum(original[n]==int(targets[i]) for n,i in enumerate(test))/len(test),holdout_predictions=original)
    pairs={}
    for i in test:
        row=records[i];pairs.setdefault((row['delay'],row['deadline'],row['rep']),{})[row['partner']]=row
    aligned=[(p[1]['original_probability']-p[0]['original_probability'])*(p[1]['label']-p[0]['label']) for p in pairs.values()]
    results['paired']=dict(holdout_pairs=len(aligned),mean_aligned_original_probability_separation=sum(aligned)/len(aligned),
        same_hard_original_predictions=sum(int((p[0]['original_probability']>=.5)==(p[1]['original_probability']>=.5)) for p in pairs.values()))
    output.mkdir(parents=True)
    tensor_path=output/'features-and-probes.pt';torch.save(tensors,tensor_path)
    report=dict(experiment='DEV-038',source=source,protocol_sha256=protocol,checkpoint_sha256=prior,
        frozen_model_hash=parameter_hash(model),tensor_sha256=hashlib.sha256(tensor_path.read_bytes()).hexdigest(),
        records=records,results=results,frozen_model_optimizer_steps=0,separately_trained_probes=True,
        scientific_test_accessed=False,capability_established=False)
    (output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v.get('holdout_accuracy') for k,v in results.items() if k!='paired'}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();run(a.input,a.output)
