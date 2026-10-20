"""Prepare real inputs, then verify exports in a fresh Torch-only process."""
import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path
import torch
from scripts.me024_coupled_research import load_source
from scripts.me022_native_readout import save_json


def probe(root,parent,previous):
    root=Path(root).resolve(); torch.set_num_threads(1)
    data=load_source(parent,previous)['dev']
    result=json.loads((root/'results.json').read_text())
    rows=[]
    with tempfile.TemporaryDirectory(prefix='me024-runtime-') as tmp:
        path=Path(tmp)/'inputs.pt'
        torch.save({k:v[:4] for k,v in data.items() if torch.is_tensor(v)},path)
        code='''
import sys,json,torch
assert not any(x=='tac_osm' or x.startswith('tac_osm.') or x=='scripts' or x.startswith('scripts.') for x in sys.modules)
torch.set_num_threads(1)
d=torch.load(sys.argv[1],weights_only=True)
m=torch.jit.load(sys.argv[2]).eval()
with torch.no_grad():
 s=torch.zeros(4,64)
 for t in range(128): s=m.stream_step(d['items'][:,t],d['metadata'][:,t],s)
 prior=(d['metadata'][:,-32:,1].sum(-1,keepdim=True)+1)/34
 a=m.readout(s,d['action'],d['duration'],prior)
 b=m(d['items'],d['metadata'],d['action'],d['duration'])
 error=float((a-b).abs().max())
 assert error<=1e-6
assert not any(x=='tac_osm' or x.startswith('tac_osm.') or x=='scripts' or x.startswith('scripts.') for x in sys.modules)
print(json.dumps({'maximum_logit_error':error,'windows':4,'tac_osm_or_scripts_imported':False}))
'''
        for row in result['models']:
            output=subprocess.check_output([sys.executable,'-c',code,str(path),str(root/row['standalone'])],text=True,cwd=tmp)
            rows.append({'arm':row['arm'],'seed':row['seed'],**json.loads(output)})
    report={'status':'PASS','processes':len(rows),'models':rows,'new_test_outcomes':0}
    save_json(root/'runtime-probe.json',report); print(json.dumps(report))


if __name__=='__main__':
 p=argparse.ArgumentParser(); p.add_argument('--root',default='results/me024')
 p.add_argument('--parent',default='results/me022'); p.add_argument('--previous',default='results/me023')
 a=p.parse_args(); probe(a.root,a.parent,a.previous)
