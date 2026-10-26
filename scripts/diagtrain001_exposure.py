"""Original FIT sampling-pool coverage; no fitting or model inference."""
import argparse
import json
import numpy as np
import torch
from scripts import me109_residual_training as run

OUT=run.ROOT/'research/mathematical-engineering/diagtrain001/EXPOSURE_COVERAGE.json'


def execute(replay=False):
    if OUT.exists() and not replay: raise RuntimeError('original output exists')
    torch.set_num_threads(1)
    d=run.fit_data();assert d['phase']=='fit'
    rows=[]
    for seed in run.SEEDS:
        p=run.OUT/'residual_guarded'/f'seed-{seed}'/'final.pt'
        cp=torch.load(p,map_location='cpu',weights_only=False);plan=cp['plan']
        assert cp['steps']==4096
        rows.append(dict(seed=seed,gap_draws=32768,general_draws=32768,
            unique_gap_origins=len(np.unique(plan[:,:8].reshape(-1,2),axis=0)),
            unique_general_origins=len(np.unique(plan[:,8:].reshape(-1,2),axis=0))))
    result=dict(scope='Original FIT eligible source origins and saved draw plans only; no fits/REPORT/protected rows',
        eligible_gap_origins=len(d['gap']),eligible_general_origins=len(d['general']),
        disjoint_source_pools=not bool(set(map(tuple,d['gap'])) & set(map(tuple,d['general']))),rows=rows)
    if replay: assert result==json.loads(OUT.read_text())
    else: run.write(OUT,result)
    print(json.dumps(result,indent=2));print('DIAGTRAIN001_EXPOSURE_REPLAY_PASS' if replay else 'DIAGTRAIN001_EXPOSURE_PASS')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--replay',action='store_true');execute(p.parse_args().replay)
