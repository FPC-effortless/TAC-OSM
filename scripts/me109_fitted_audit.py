"""Original trained-model graph and read-only proposal audit. No optimizer step."""
import argparse
import json

import numpy as np
import torch

from scripts import me109_residual_training as run
from scripts.me109_graph_reference import serial
from scripts.me108_optimizer_reference import step as ref_step
from tac_osm.role_optimizer import BlockProtectedAdamW

OUT=run.OUT/'FITTED_GRAPH_AUDIT.json'


def audit(replay=False, prefit=False):
    output = run.DOC/'PREFIT_GRAPH_AUDIT.json' if prefit else OUT
    if not replay and output.exists():
        raise RuntimeError('original graph audit exists; no overwrite')
    index = dict(models=[dict(seed=s) for s in run.SEEDS]) if prefit else run.budget_audit()
    torch.set_num_threads(1)
    d=run.fit_data()
    rng=np.random.default_rng(109999)
    origins=np.asarray(d['general'])[rng.integers(len(d['general']),size=4)]
    original=run.tensor_batch(d,origins,64,rng)
    rawscale=np.nanstd(d['raw'][...,:6].astype(np.float64),axis=(0,1)).clip(1e-4)
    rows=[]
    for identity in index['models']:
        seed=identity['seed'];path=run.OUT/'residual_guarded'/f'seed-{seed}'/'final.pt'
        if prefit:
            _, parent, _ = run.parent_artifacts(seed)
            initialized = run.initialize(d, seed)
            run.load_parent(initialized, parent)
            # Only this disposable clone receives nonzero output maps. Original
            # initialized weights, parents, plans and checkpoints stay untouched.
            import copy
            cloned = copy.deepcopy(initialized)
            generator = torch.Generator().manual_seed(seed + 109000)
            with torch.no_grad():
                for role in ('current', 'width', 'median'):
                    head = getattr(cloned, role + '_residual').output
                    for parameter in head.parameters():
                        parameter.copy_(torch.randn(parameter.shape, generator=generator) * .025)
            cp = dict(model=cloned.state_dict())
        else:
            cp=torch.load(path,map_location='cpu',weights_only=False)
            if cp['steps']!=4096 or cp['seed']!=seed or cp['architecture']!='residual_guarded':
                raise AssertionError('checkpoint identity/steps mismatch')
            if len(cp['optimizer']['state'])!=45 or {int(v['step']) for v in cp['optimizer']['state'].values()}!={4096}:
                raise AssertionError('45 original moments must be at 4096')
        for dtype,rtol,atol in ((torch.float32,2e-5,2e-6),(torch.float64,1e-9,1e-10)):
            model=run.initialize(d,seed).to(dtype)
            model.load_state_dict(cp['model'],strict=True)
            versions=model.versions()
            x=[v.to(dtype).detach().clone() if v.is_floating_point() else v.clone() for v in original]
            xr=[v.detach().clone() for v in x]
            for i in (0, 2, 3):
                x[i].requires_grad_(); xr[i].requires_grad_()
            if len(tuple(model.parameters())) != 45:
                raise AssertionError('audit requires all 45 parameter tensors')
            scale=torch.tensor(rawscale,dtype=dtype)
            forecast,state,trace=model(*x[:6],return_trace=True)
            mu,left,right,ss,sm,es,memory,rt=serial(model,*xr[:5])
            errors={}
            for label,a,b in zip(('location','left','right','source_sum','source_mass','evidence_sum','memory'),
                    (forecast.log_location,forecast.left_scale,forecast.right_scale,state.source_sum,state.source_mass,state.evidence_sum,state.memory),
                    (mu,left,right,ss,sm,es,memory)):
                torch.testing.assert_close(a,b,rtol=rtol,atol=atol,msg=label)
                errors[label]=float((a-b).detach().abs().max())
            loss,parts=run.role_objective(model,forecast,trace,x,scale)
            params=tuple(model.parameters())
            point=torch.autograd.grad(.5*parts['point'],params,retain_graph=True,allow_unused=True)
            inputs=tuple(x[i] for i in (0,2,3))
            ref_inputs=tuple(xr[i] for i in (0,2,3))
            gradients=torch.autograd.grad(loss,params+inputs,allow_unused=True)
            expected,refparts=run.role_objective(model,type(forecast)(mu,left,right),rt,xr,scale)
            for key in parts:
                torch.testing.assert_close(parts[key],refparts[key],rtol=rtol,atol=atol)
            expected_grads=torch.autograd.grad(expected,params+ref_inputs,allow_unused=True)
            max_gradient=0.
            for p,a,b in zip(params,gradients,expected_grads):
                a=torch.zeros_like(p) if a is None else a
                b=torch.zeros_like(p) if b is None else b
                assert torch.isfinite(a).all()
                torch.testing.assert_close(a,b,rtol=rtol,atol=atol)
                max_gradient=max(max_gradient,float((a-b).abs().max()))
                p.grad=a.detach().clone()
            input_errors={}
            for label,p,a,b in zip(('normalized_values','ages','calendar'),inputs,gradients[45:],expected_grads[45:]):
                a=torch.zeros_like(p) if a is None else a
                b=torch.zeros_like(p) if b is None else b
                assert torch.isfinite(a).all() and torch.isfinite(b).all()
                torch.testing.assert_close(a,b,rtol=rtol,atol=atol,msg=label)
                input_errors[label]=float((a-b).abs().max())
            if prefit:
                assert model.versions()==versions
                rows.append(dict(seed=seed,dtype=str(dtype),rtol=rtol,atol=atol,forward_errors=errors,
                    max_gradient_abs_error=max_gradient,input_gradient_errors=input_errors,
                    parameter_tensors=45,nonzero_residual_clone_only=True,no_optimizer_created=True,
                    no_optimizer_step=True))
                continue
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
            for p,before,old_state in zip(params,old,states):
                torch.testing.assert_close(p,before,rtol=0,atol=0)
                for key in ('step','exp_avg','exp_avg_sq'):
                    torch.testing.assert_close(opt.state[p][key],old_state[key],rtol=0,atol=0)
            assert all(r['applied_dot']<=0 for r in receipt)
            assert model.versions()==versions and run.sha(path)==identity['checkpoint_sha256']
            assert {int(v['step']) for v in opt.state.values()}=={4096}
            rows.append(dict(seed=seed,dtype=str(dtype),rtol=rtol,atol=atol,forward_errors=errors,
                             max_gradient_abs_error=max_gradient,input_gradient_errors=input_errors,parameter_tensors=45,read_only_proposal_and_moments_bit_exact=True,
                             proposal_computed_point_dots_nonpositive=True,no_optimizer_step=True))
    result=dict(models=5,checks=rows,no_optimizer_updates=True,fit_only=True,report_or_protected_access=False)
    result["prefit"] = prefit
    result["no_optimizer_created"] = prefit
    if replay:
        expected=json.loads(output.read_text())
        assert [(r['seed'],r['dtype']) for r in rows]==[(r['seed'],r['dtype']) for r in expected['checks']]
        # All original strict tensor assertions run above; hardware-dependent
        # diagnostic error maxima need not themselves be identical.
        print('ME109_FITTED_GRAPH_AND_PROPOSAL_REPLAY_PASS')
    else:
        run.write(output,result)
        print('ME109_PREFIT_GRAPH_PASS' if prefit else 'ME109_FITTED_GRAPH_AND_PROPOSAL_PASS')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('prefit','run','replay'))
    mode=p.parse_args().mode
    audit(replay=mode=='replay',prefit=mode=='prefit')
