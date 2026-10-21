"""Replay recorded ME026 exports, validate timings and probe fresh Torch runtimes."""
from __future__ import annotations
import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path
import numpy as np
import torch
from scripts.me025_recorded_audit import C,audit as original_audit
from scripts.me025_real_transformer_parity import prob_scores
from scripts.me023_predictive_research import sha,load_source
from scripts.me022_native_readout import save_json


def audit(root,original,parent,archive):
    root=Path(root); report=json.loads((root/'runtime-results.json').read_text())
    assert report['contract']==C and report['new_fits']==0 and len(report['models'])==6
    assert {(m['arm'],m['seed']) for m in report['models']}=={(a,s) for a in C['native_arms'] for s in C['seeds']}
    for m in report['models']: assert sha(root/m['standalone'])==m['standalone_sha256']
    original_audit(original,parent,root,archive)
    data=load_source(parent)['dev']; torch.set_num_threads(1)
    records=[]
    with tempfile.TemporaryDirectory(prefix='me026-fresh-runtime-') as tmp:
        inputs=Path(tmp)/'genuine-inputs.pt'
        torch.save({k:v[:4] for k,v in data.items() if torch.is_tensor(v)},inputs)
        code='''
import sys,json,torch
assert not any(n=='tac_osm' or n.startswith('tac_osm.') or n=='scripts' or n.startswith('scripts.') for n in sys.modules)
torch.set_num_threads(1)
d=torch.load(sys.argv[1],weights_only=True);m=torch.jit.load(sys.argv[2]).eval()
with torch.no_grad():
 state=torch.zeros(4,m.state_width)
 for t in range(128):state=m.stream_step(d['items'][:,t],d['metadata'][:,t],state)
 prior=(d['metadata'][:,-32:,1].sum(-1,keepdim=True)+1)/34
 a=m.readout(state,d['action'],d['duration'],prior)
 b=m(d['items'],d['metadata'],d['action'],d['duration'])
 le=float((a-b).abs().max());pe=float((a.sigmoid()-b.sigmoid()).abs().max())
 assert le<=1e-5 and pe<=1e-6
assert not any(n=='tac_osm' or n.startswith('tac_osm.') or n=='scripts' or n.startswith('scripts.') for n in sys.modules)
print(json.dumps({'logit_error':le,'probability_error':pe,'tac_osm_or_scripts_imported':False}))
'''
        for m in report['models']:
            net=torch.jit.load(str(root/m['standalone'])).eval(); p=prob_scores(net,data)
            old=np.load(root/f"{m['arm']}-seed{m['seed']}-compiled-predictions.npy")
            e=float(np.max(np.abs(p-old))); assert e<=C['runtime_probability_atol']
            original_p=np.load(Path(original)/f"{m['arm']}-seed{m['seed']}-dev-predictions.npy")
            oe=float(np.max(np.abs(p-original_p))); assert oe<=C['runtime_probability_atol']
            assert [t['batch'] for t in m['timings']]==C['benchmark_batches']
            for t in m['timings']:
                for name,samples in t['samples'].items():
                    assert len(samples)==C['benchmark_repeats']
                    for field in ['wall_seconds','cpu_seconds']:
                        values=[v[field] for v in samples];assert min(values)>0
                        assert float(np.median(values))==t['medians'][name][field]
                speed=t['medians']['eager']['wall_seconds']/t['medians']['compiled']['wall_seconds']
                assert speed==t['equal_work_full_window_speedup']
            fresh=json.loads(subprocess.check_output([sys.executable,'-c',code,str(inputs),str((root/m['standalone']).resolve())],text=True,cwd=tmp))
            records.append({'arm':m['arm'],'seed':m['seed'],'maximum_replay_probability_error':e,
                'maximum_original_probability_error':oe,'fresh_Torch_only':fresh})
    speeds=[m['timings'][-1]['equal_work_full_window_speedup'] for m in report['models']]
    assert all(v>=1/.9 for v in speeds)==report['primary_engineering_gate_passed']
    result={'status':'PASS','models':records,'native_predictions_per_path':6*len(data['y']),
        'original_neural_predictions_per_path':9*len(data['y']),'original_neural_checkpoints':9,
        'native_standalone_exports':6,'fresh_Torch_only_processes':6,'timing_medians_and_gates_reconstructed':True,
        'new_timing_runs':0,'new_fits':0,'new_test_outcomes':0}
    save_json(root/'runtime-replay.json',result);print(json.dumps(result));return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',default='results/me026')
    p.add_argument('--original',default='/workspace/me025-workflow-original/results/me025')
    p.add_argument('--parent',default='results/me022');p.add_argument('--archive',default='/workspace/me025-original-workflow.zip')
    a=p.parse_args();audit(a.root,a.original,a.parent,a.archive)
