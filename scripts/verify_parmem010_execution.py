#!/usr/bin/env python3
"""Cross-repository full prepass/scan/decoder verification. No fit or data access."""
import argparse
from dataclasses import replace
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/'src'))
from scripts.verify_parmem010_math import (
    model_fixture,inputs,weights,source_pins,batch,Checks,as_numpy,SOURCE_SHA,
)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pnds-root',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args()
    sys.path.insert(0,str(args.pnds_root.resolve()))
    from foundation.v1.source_aligned_model_math import SourceAlignedModelReference
    from foundation.v1.source_aligned_execution_math import prepass_and_scan
    torch.set_num_threads(1)
    checks=Checks()
    pins=source_pins()
    for dtype in (torch.float32,torch.float64):
        for t,pattern in [(1,'none'),(3,'mixed'),(7,'all'),(16,'mixed'),(33,'none'),(129,'mixed')]:
            model=model_fixture(11010+t,dtype)
            data=inputs(t,dtype,pattern)
            arrays=tuple(as_numpy(v) for v in data)
            reference=SourceAlignedModelReference(weights(model))
            with torch.no_grad():
                expected,trace=batch(model,data)
                actual,scanned,states=prepass_and_scan(reference,arrays)
            atol,rtol=((2e-5,2e-4) if dtype==torch.float32 else (2e-12,2e-10))
            label=f'{str(dtype).split(".")[-1]}/T{t}/{pattern}'
            checks.close(label+'/whole_graph',expected,actual,atol,rtol)
            checks.close(label+'/native_prefixes',trace['memory'],scanned[:,:,0],atol,rtol)
            checks.close(label+'/aligned_prefixes',trace['aligned_memory'],
                         scanned[:,:,1:].reshape(2,t,2,16,8),atol,rtol)
            checks.close(label+'/native_step_writes',
                         np.stack([s.memory for s in states],axis=1),scanned[:,:,0],atol,rtol)
            checks.close(label+'/aligned_step_writes',
                         np.stack([s.aligned for s in states],axis=1),
                         scanned[:,:,1:].reshape(2,t,2,16,8),atol,rtol)
    # Pinned Torch 2.14.1 backend: clamp_min chooses its constant branch at ties.
    x=torch.tensor(0.,dtype=torch.float64,requires_grad=True)
    clamp_gradient=torch.autograd.grad(x.clamp_min(0.),x)[0].item()
    assert clamp_gradient==0.
    # Normalization is smooth at a zero key due to the positive epsilon floor.
    key=torch.zeros(8,dtype=torch.float64,requires_grad=True)
    jac=torch.autograd.functional.jacobian(
        lambda z:torch.nn.functional.normalize(z,dim=-1,eps=1e-6),key)
    checks.close('normalizer_zero_jacobian',jac,torch.eye(8,dtype=torch.float64)*1e6,0.,0.)
    # All-finite inputs/weights alone do NOT establish a numerical domain:
    # normalization of an overflowed learned address can produce nonfinite P/Q.
    model=model_fixture(11011,torch.float32)
    with torch.no_grad():model.source_key_rotations.fill_(torch.finfo(torch.float32).max)
    assert all(torch.isfinite(p).all().item() for p in model.parameters())
    fixture=inputs(1,torch.float32,'all')
    rejected=False
    try:
        with torch.no_grad():batch(model,fixture)
    except ValueError as error:
        rejected='nonfinite' in str(error)
    assert rejected,'finite-parameter overflow witness unexpectedly failed'
    # A valid two-event physical clock already separates FP32 from exact reals.
    clock=torch.tensor(2**24,dtype=torch.float32)
    assert float(clock+1.)==float(clock)
    healthy=model_fixture(11012,torch.float32)
    x,a,dt,mask,owners,events=inputs(1,torch.float32,'all')
    state=healthy.init_stream(owners)
    with torch.no_grad():
        for offset,delta in [(0,float(2**24)),(1,1.)]:
            _,state=healthy.stream_step(state,x[:,0],a[:,0],
                elapsed_dt=torch.full((2,),delta),write_mask=mask[:,0],
                source_ids=owners,event_index=events[:,0]+offset)
    assert torch.equal(state.native.elapsed_clock,torch.full((2,),float(2**24)))
    receipt=dict(schema='parmem010.complete_execution_checks.v1',source_commit=SOURCE_SHA,
                 source_sha256=pins,status='PASS_NUMERICAL_NOT_BACKEND_FORMAL_PROOF',
                 optimizer_steps=0,checks=checks.results,
                 torch_version=torch.__version__,numpy_version=np.__version__,device='cpu',
                 reference_sha256={path:hashlib.sha256((args.pnds_root/path).read_bytes()).hexdigest()
                    for path in ['foundation/v1/source_aligned_model_math.py',
                                 'foundation/v1/source_aligned_execution_math.py']},
                 mathematical_boundary_witnesses={
                     'clamp_tie_torch_selected_derivative':clamp_gradient,
                     'zero_key_derivative_is_identity_over_epsilon':True,
                     'all_finite_parameters_do_not_prevent_overflow':rejected,
                     'fp32_clock_2pow24_plus_one_stagnates':True})
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps({'status':receipt['status'],'comparisons':len(checks.results),
                      'boundary_witnesses':receipt['mathematical_boundary_witnesses']}))


if __name__=='__main__':main()
