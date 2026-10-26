"""Original trained-model graph and read-only proposal audit. No optimizer step."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from scripts import me108_role_training as run
from scripts.me107_graph_reference import serial
from scripts.me108_optimizer_reference import step as ref_step
from tac_osm.role_optimizer import BlockProtectedAdamW

OUT=run.OUT/'FITTED_GRAPH_AUDIT.json'


def audit(replay=False):
    index=run.budget_audit()
    torch.set_num_threads(1)
    d=run.fit_data()
    rng=np.random.default_rng(108999)
    origins=np.asarray(d['general'])[rng.integers(len(d['general']),size=4)]
    original=run.tensor_batch(d,origins,64,rng)
    rawscale=np.nanstd(d['raw'][...,:6].astype(np.float64),axis=(0,1)).clip(1e-4)
    rows=[]
    for identity in index['models']:
        seed=identity['seed'];path=run.OUT/'role_guarded'/f'seed-{seed}'/'final.pt'
        cp=torch.load(path,map_location='cpu',weights_only=False)
        if cp['steps']!=4096 or cp['seed']!=seed or cp['architecture']!='role_guarded':
            raise AssertionError('checkpoint identity/steps mismatch')
        if len(cp['optimizer']['state'])!=39 or {int(v['step']) for v in cp['optimizer']['state'].values()}!={4096}:
            raise AssertionError('39original moments must be at4096')
        for dtype,rtol,atol in ((torch.float32,2e-5,2e-6),(torch.float64,1e-9,1e-10)):
            model=run.initialize(d,seed).to(dtype)
            model.load_state_dict(cp['model'],strict=True)
            versions=model.versions()
            x=[v.to(dtype) if v.is_floating_point() else v for v in original]
            scale=torch.tensor(rawscale,dtype=dtype)
            forecast,state,trace=model(*x[:6],return_trace=True)
            mu,left,right,ss,sm,es,memory,rt=serial(model,*x[:5])
            errors={}
            for label,a,b in zip(('location','left','right','source_sum','source_mass','evidence_sum','memory'),
                    (forecast.log_location,forecast.left_scale,forecast.right_scale,state.source_sum,state.source_mass,state.evidence_sum,state.memory),
                    (mu,left,right,ss,sm,es,memory)):
                torch.testing.assert_close(a,b,rtol=rtol,atol=atol,msg=label)
                errors[label]=float((a-b).detach().abs().max())
            loss,parts=run.role_objective(model,forecast,trace,x,scale)
            params=tuple(model.parameters())
            point=torch.autograd.grad(.5*parts['point'],params,retain_graph=True,allow_unused=True)
            gradients=torch.autograd.grad(loss,params,allow_unused=True)
            expected,_=run.role_objective(model,type(forecast)(mu,left,right),rt,x,scale)
            expected_grads=torch.autograd.grad(expected,params,allow_unused=True)
            max_gradient=0.
            for p,a,b in zip(params,gradients,expected_grads):
                a=torch.zeros_like(p) if a is None else a
                b=torch.zeros_like(p) if b is None else b
                assert torch.isfinite(a).all()
                torch.testing.assert_close(a,b,rtol=rtol,atol=atol)
                max_gradient=max(max_gradient,float((a-b).abs().max()))
                p.grad=a.detach().clone()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            point=[torch.zeros_like(p) if g is None else g.detach().clone() for p,g in zip(params,point)]
            _,blockmap=run.blocks(model);blocks=list(blockmap.values())
            opt=BlockProtectedAdamW(run.parameter_groups(model),blocks=blocks)
            opt.load_state_dict(cp['optimizer'])
            old=[p.detach().clone() for p in params]
            configs=[{k:g[k] for k in ('lr','betas','eps','weight_decay')} for g in opt.param_groups for _ in g['params']]
            states=[{k:v.detach().clone() for k,v in opt.state[p].items()} for p in params]
            wanted,wanted_state=ref_step(old,[p.grad for p in params],point,states,configs,blocks)
            proposed,proposed_state,receipt=opt.propose(point)  # NO step or mutation
            for a,b,sa,sb in zip(proposed,wanted,proposed_state,wanted_state):
                torch.testing.assert_close(a,b,rtol=0,atol=0)
                for key in ('step','exp_avg','exp_avg_sq'):torch.testing.assert_close(sa[key],sb[key],rtol=0,atol=0)
            assert all(r['applied_dot']<=0 for r in receipt)
            assert model.versions()==versions and run.sha(path)==identity['checkpoint_sha256']
            assert {int(v['step']) for v in opt.state.values()}=={4096}
            rows.append(dict(seed=seed,dtype=str(dtype),rtol=rtol,atol=atol,forward_errors=errors,
                             max_gradient_abs_error=max_gradient,read_only_proposal_and_moments_bit_exact=True,
                             proposal_computed_point_dots_nonpositive=True,no_optimizer_step=True))
    result=dict(models=5,checks=rows,no_optimizer_updates=True,fit_only=True,report_or_protected_access=False)
    if replay:
        expected=json.loads(OUT.read_text())
        assert [(r['seed'],r['dtype']) for r in rows]==[(r['seed'],r['dtype']) for r in expected['checks']]
        # All original strict tensor assertions run above; hardware-dependent
        # diagnostic error maxima need not themselves be identical.
        print('ME108_FITTED_GRAPH_AND_PROPOSAL_REPLAY_PASS')
    else:
        if OUT.exists():raise RuntimeError('original fitted audit exists; replay only')
        run.write(OUT,result)
        print('ME108_FITTED_GRAPH_AND_PROPOSAL_PASS')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('run','replay'))
    audit(p.parse_args().mode=='replay')
