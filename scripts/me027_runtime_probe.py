"""Fresh Torch-only probes of all registered primary exports; no fitting."""
import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path
import torch
from scripts.me027_state_research import C
from scripts.me027_audit import selected
from scripts.me023_predictive_research import sha,load_source
from scripts.me022_native_readout import save_json


def probe(root,parent):
    root=Path(root).resolve();r=json.loads((root/'results.json').read_text());primary=[selected(m,'cpu20') for m in r['models']]
    assert len(primary)==12
    for q in primary:assert sha(root/q['standalone'])==q['standalone_sha256']
    data=load_source(parent)['dev'];torch.set_num_threads(1);rows=[]
    code='''
import sys,json,torch,numpy as np
assert not any(n=='tac_osm' or n.startswith('tac_osm.') or n=='scripts' or n.startswith('scripts.') for n in sys.modules)
torch.set_num_threads(1)
d=torch.load(sys.argv[1],weights_only=True);m=torch.jit.load(sys.argv[2]).eval()
with torch.no_grad():p=m(d['items'],d['metadata'],d['action'],d['duration']).sigmoid().flatten().numpy()
old=np.load(sys.argv[3])[:4];e=float(np.max(np.abs(p-old)))
assert e<=1e-6
assert not any(n=='tac_osm' or n.startswith('tac_osm.') or n=='scripts' or n.startswith('scripts.') for n in sys.modules)
print(json.dumps({'maximum_probability_error':e,'tac_osm_or_scripts_imported':False,'real_windows':4}))
'''
    with tempfile.TemporaryDirectory(prefix='me027-runtime-') as temp:
        inputs=Path(temp)/'inputs.pt';torch.save({k:v[:4] for k,v in data.items() if torch.is_tensor(v)},inputs)
        for m,q in zip(r['models'],primary):
            output=subprocess.check_output([sys.executable,'-c',code,str(inputs),str(root/q['standalone']),str(root/q['prediction'])],text=True,cwd=temp)
            rows.append({'arm':m['arm'],'seed':m['seed'],**json.loads(output)})
    report={'status':'PASS','processes':len(rows),'models':rows,'new_fits':0,'new_test_outcomes':0}
    save_json(root/'runtime-probe.json',report);print(json.dumps(report));return report


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',default='results/me027');p.add_argument('--parent',default='results/me022')
    a=p.parse_args();probe(a.root,a.parent)
