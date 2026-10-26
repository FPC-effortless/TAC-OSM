"""Restore all original ME110 checkpoint stages and fixed FIT monitoring arrays.

No optimizer creation or updates, no REPORT or protected observations.
"""
import argparse
import json
import numpy as np
import torch
from scripts import me110_typed_training as run
from scripts.me109_graph_reference import serial
from scripts.me110_objective import losses

OUT=run.OUT/'CHECKPOINT_REPLAY_AUDIT.json'


def execute(replay=False):
    if OUT.exists() and not replay: raise RuntimeError('original audit exists')
    torch.set_num_threads(1)
    index=run.budget_audit();d=run.fit_data()
    monitor=np.load(run.DOC/'FIT_PLANS.npz')['monitor'];rows=[];scalar_replays=0
    for identity in index['models']:
        seed=identity['seed'];folder=run.OUT/run.ARCH/f'seed-{seed}'
        for step in (0,*run.CHECKPOINTS):
            model=run.initialize(d,seed)
            if step==0:
                _,cp,_=run.parent_artifacts(seed);run.load_parent(model,cp)
                checkpoint_sha=run.sha(run.parent_artifacts(seed)[0])
            else:
                path=folder/('final.pt' if step==run.STEPS else f'checkpoint-{step}.pt')
                checkpoint_sha=run.sha(path);cp=torch.load(path,map_location='cpu',weights_only=False)
                assert cp['seed']==seed and cp['steps']==step
                assert {int(s['step']) for s in cp['optimizer']['state'].values()}=={step}
                assert len(cp['optimizer']['state'])==45
                model.load_state_dict(cp['model'],strict=True)
            run.inventory(model);before=model.versions()
            original=np.load(folder/f'monitor-{step}.npz');max_error=0.
            with torch.no_grad():
                for name,origins in [('gap',monitor[:64]),('general',monitor[64:])]:
                    qs=[]
                    for i in range(0,len(origins),32):
                        x=[torch.from_numpy(v) for v in run.data.batch(d,origins[i:i+32],256)]
                        forecast,_=model(*x[:6]);qs.append(model.raw_quantiles(forecast).numpy())
                    actual=np.concatenate(qs);expected=original[name+'_quantiles']
                    np.testing.assert_allclose(actual,expected,rtol=1e-5,atol=1e-6)
                    max_error=max(max_error,float(np.abs(actual-expected).max()));scalar_replays+=actual.size
            assert model.versions()==before
            rows.append(dict(seed=seed,step=step,checkpoint_sha256=checkpoint_sha,monitor_scalar_max_abs_error=max_error,
                parameters=104663,no_optimizer_created=True,no_new_model_fit=True))
        # Actual fitted native scan versus independent serial Python equations.
        path=folder/'final.pt';cp=torch.load(path,map_location='cpu',weights_only=False)
        model=run.initialize(d,seed);model.load_state_dict(cp['model'],strict=True)
        x=run.tensor_batch(d,monitor[64:68],64,np.random.default_rng(110999))
        with torch.no_grad():
            actual,state,_=model(*x[:6],return_trace=True)
            mu,left,right,ss,sm,es,memory,_=serial(model,*x[:5])
            for a,b in zip((actual.log_location,actual.left_scale,actual.right_scale,state.source_sum,state.source_mass,state.evidence_sum,state.memory),(mu,left,right,ss,sm,es,memory)):
                torch.testing.assert_close(a,b,rtol=2e-5,atol=2e-6)
        for dtype,rtol,atol in [(torch.float32,2e-5,2e-6),(torch.float64,1e-9,1e-10)]:
            m=run.initialize(d,seed).to(dtype);m.load_state_dict(cp['model'],strict=True)
            actual_inputs=[v.to(dtype).detach().clone() if v.is_floating_point() else v.clone() for v in x]
            reference_inputs=[v.detach().clone() for v in actual_inputs]
            for i in (0,2,3):actual_inputs[i].requires_grad_();reference_inputs[i].requires_grad_()
            f,_,trace=m(*actual_inputs[:6],return_trace=True)
            mu,left,right,*_,reftrace=serial(m,*reference_inputs[:5])
            scale=run.raw_scale(d).to(dtype)
            p,u,_=losses(m,f,trace,actual_inputs,scale)
            rp,ru,_=losses(m,type(f)(mu,left,right),reftrace,reference_inputs,scale)
            params=tuple(m.parameters())
            gradients=torch.autograd.grad(p+u,params+tuple(actual_inputs[i] for i in (0,2,3)),allow_unused=True)
            reference_gradients=torch.autograd.grad(rp+ru,params+tuple(reference_inputs[i] for i in (0,2,3)),allow_unused=True)
            for original,reference in zip(gradients,reference_gradients):
                assert (original is None)==(reference is None)
                if original is not None:torch.testing.assert_close(original,reference,rtol=rtol,atol=atol)
        assert run.sha(path)==identity['checkpoint_sha256']
    result=dict(status='ORIGINAL_ME110_CHECKPOINT_MONITOR_REPLAY_PASS',scope='FIT_ONLY_NO_UPDATES',
        checkpoint_views=20,forecast_scalars=scalar_replays,checks=rows,original_final_serial_reference_models=5,final_forward_backward_fp32_fp64_checks=10,
        protected_or_report_access=False,whole_neural_ieee_proof=False)
    if replay: run._compare_numeric(result,json.loads(OUT.read_text()))
    else: run.write(OUT,result)
    print('ME110_CHECKPOINT_MONITOR_REPLAY_PASS',scalar_replays)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--replay',action='store_true');execute(p.parse_args().replay)
