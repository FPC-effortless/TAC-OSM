"""Additional normalized-input gradient check on frozen original models."""
import argparse
import json
import numpy as np
import torch

from scripts import me108_role_training as run
from scripts.me107_graph_reference import serial

OUT=run.OUT/'FITTED_INPUT_AUDIT.json'


def audit(replay=False):
    torch.set_num_threads(1)
    index=run.budget_audit();d=run.fit_data()
    rng=np.random.default_rng(108998)
    origins=np.asarray(d['general'])[rng.integers(len(d['general']),size=4)]
    batch=run.tensor_batch(d,origins,64,rng)
    rows=[]
    for row in index['models']:
        path=run.OUT/'role_guarded'/f"seed-{row['seed']}"/'final.pt'
        cp=torch.load(path,map_location='cpu',weights_only=False)
        for dtype,rtol,atol in ((torch.float32,2e-5,2e-6),(torch.float64,1e-9,1e-10)):
            model=run.initialize(d,row['seed']).to(dtype);model.load_state_dict(cp['model'])
            x=[t.to(dtype) if t.is_floating_point() else t for t in batch]
            for i in (0,2,3):x[i]=x[i].clone().requires_grad_(True)
            versions=model.versions()
            f,_=model(*x[:6]);mu,l,r,*_=serial(model,*x[:5])
            la=f.log_location.square().sum()+f.left_scale.square().sum()+f.right_scale.square().sum()
            lb=mu.square().sum()+l.square().sum()+r.square().sum()
            inputs=[x[i] for i in (0,2,3)]
            ga=torch.autograd.grad(la,inputs);gb=torch.autograd.grad(lb,inputs)
            errors={}
            for name,a,b in zip(('normalized_values','normalized_ages','clock_features'),ga,gb):
                assert torch.isfinite(a).all()
                torch.testing.assert_close(a,b,rtol=rtol,atol=atol,msg=name)
                errors[name]=float((a-b).abs().max())
            assert model.versions()==versions and run.sha(path)==row['checkpoint_sha256']
            rows.append(dict(seed=row['seed'],dtype=str(dtype),rtol=rtol,atol=atol,errors=errors))
    result=dict(fit_only=True,no_optimizer_updates=True,scope='normalized permitted inputs, not raw physical/causal intervention derivatives',checks=rows)
    if replay:
        old=json.loads(OUT.read_text());assert [(r['seed'],r['dtype']) for r in rows]==[(r['seed'],r['dtype']) for r in old['checks']]
        print('ME108_INPUT_GRADIENT_REPLAY_PASS')
    else:
        if OUT.exists():raise RuntimeError('original input audit exists; replay only')
        run.write(OUT,result);print('ME108_INPUT_GRADIENT_PASS')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('run','replay'));audit(p.parse_args().mode=='replay')
