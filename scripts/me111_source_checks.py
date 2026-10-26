"""Actual five-parent-source forward/backward parity, no optimizer or REPORT."""
import json
import numpy as np
import torch
from scripts import me111_conditional_training as run
from scripts.me111_graph_reference import serial
from scripts.me111_objective import losses


def execute():
    torch.set_num_threads(1);d=run.fit_data();monitor=np.load(run.DOC/'FIT_PLANS.npz')['monitor'];rows=[]
    for seed in run.SEEDS:
        _,cp,_=run.parent_artifacts(seed)
        x=run.tensor_batch(d,monitor[64:68],64,np.random.default_rng(111999))
        for dtype,rtol,atol in [(torch.float32,2e-5,2e-6),(torch.float64,1e-9,1e-10)]:
            m=run.initialize(d,seed).to(dtype);m.load_state_dict(cp['model'],strict=True)
            a=[v.to(dtype).detach().clone() if v.is_floating_point() else v.clone() for v in x]
            b=[v.detach().clone() for v in a]
            for i in (0,2,3):a[i].requires_grad_();b[i].requires_grad_()
            f,s,t=m(*a[:6],return_trace=True);mu,l,r,ss,sm,es,memory,rt=serial(m,*b[:5])
            for aa,bb in zip((f.log_location,f.left_scale,f.right_scale,s.source_sum,s.source_mass,s.evidence_sum,s.memory),(mu,l,r,ss,sm,es,memory)):
                torch.testing.assert_close(aa,bb,rtol=rtol,atol=atol)
            p,u,_=losses(m,f,t,a,run.raw_scale(d).to(dtype));rp,ru,_=losses(m,type(f)(mu,l,r),rt,b,run.raw_scale(d).to(dtype))
            params=tuple(m.parameters())
            ga=torch.autograd.grad(p+u,params+tuple(a[i] for i in (0,2,3)),allow_unused=True)
            gb=torch.autograd.grad(rp+ru,params+tuple(b[i] for i in (0,2,3)),allow_unused=True)
            for aa,bb in zip(ga,gb):
                assert (aa is None)==(bb is None)
                if aa is not None:torch.testing.assert_close(aa,bb,rtol=rtol,atol=atol)
            rows.append(dict(seed=seed,dtype=str(dtype),rtol=rtol,atol=atol,parameter_tensors=45,input_gradients=3,forward_backward='PASS'))
    run.write(run.DOC/'SOURCE_PARITY.json',dict(checks=rows,no_optimizer_created=True,no_report_or_protected_access=True,full_neural_lean_ieee_proof=False))
    print('ME111_INITIAL_SOURCE_PARITY_PASS',len(rows))
if __name__=='__main__':execute()
