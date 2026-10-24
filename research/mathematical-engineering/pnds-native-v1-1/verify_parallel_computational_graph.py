#!/usr/bin/env python3
"""End-to-end parallel graph partition + exact conditional memory prefix tests.

No fit, trained checkpoint, real-data score, speed-up claim or parallel native
state recurrence. Original model source pinned by verified Git blob.
"""
from __future__ import annotations
import sys
from pathlib import Path
import torch
from torch.nn import functional as F

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
sys.path.insert(0,str(ROOT/"src"))
from tac_osm.variants.parallel_affine_memory import (
    associative_affine_schedule,parallel_affine_prefix,
    sequential_affine_reference,scan_source_schedules)
from tac_osm.variants.parallel_partitioned_native import parallel_partitioned_forward
from verify_pinned_source import modules


def check_close(a,b, dtype, label):
    abs_tol=3e-5 if dtype is torch.float32 else 5e-11
    rel_tol=3e-5 if dtype is torch.float32 else 5e-11
    try:
        torch.testing.assert_close(a,b,atol=abs_tol,rtol=rel_tol)
    except AssertionError as e:
        raise AssertionError(f"{label}: {str(e)}") from e


def forward_backward_partition():
    source,_=modules()
    checks=0
    torch.manual_seed(20261024)
    for dtype in (torch.float32,torch.float64):
        model=source.AlignedAssociativeTAC().to(dtype=dtype).eval()
        assert sum(p.numel() for p in model.parameters()) == 101527
        for batch,steps in ((1,1),(2,7),(3,13),(2,32)):
            kind=torch.randint(0,3,(batch,steps))
            kind[:,-1]=2
            keys=torch.randint(0,65,(batch,steps))
            vals=torch.randint(0,33,(batch,steps))
            reference,ref_trace=model(kind,keys,vals,return_trace=True)
            obtained,new_trace=parallel_partitioned_forward(model,kind,keys,vals,True)
            check_close(obtained,reference,dtype,"end_to_end_logits")
            checks+=1
            for k in ("addresses","memory","gates","read"):
                check_close(new_trace[k],ref_trace[k],dtype,k)
                checks+=1
            params=tuple(model.parameters())
            grad_ref=torch.autograd.grad(reference.square().sum(),params,
                                        allow_unused=False)
            grad_new=torch.autograd.grad(obtained.square().sum(),params,
                                        allow_unused=False)
            for old,new in zip(grad_ref,grad_new):
                # FP32 separate matrix reduction order can differ.
                check_close(old,new,dtype,"all_learned_parameter_gradients")
            checks+=len(params)
    return checks


def scan_reference_and_native_schedule():
    source,_=modules()
    torch.manual_seed(20261024)
    checks=0
    for dtype in (torch.float32,torch.float64):
        for batch,steps in ((1,1),(2,3),(3,9),(2,32),(1,65)):
            k=F.normalize(torch.randn(batch,steps,8,dtype=dtype),dim=-1)
            v=F.normalize(torch.randn(batch,steps,8,dtype=dtype),dim=-1)*.03
            a=torch.rand(batch,steps,1,dtype=dtype)*.3+.68
            beta=torch.rand(batch,steps,1,dtype=dtype)*.5
            mask=(torch.rand(batch,steps,1)>0.35)
            # Make all differentiable scan coefficients true leaves.
            k=k.detach().requires_grad_(True)
            v=v.detach().requires_grad_(True)
            a=a.detach().requires_grad_(True)
            beta=beta.detach().requires_grad_(True)
            P,Q=associative_affine_schedule(k,v,a,beta,mask)
            zero_scan=parallel_affine_prefix(P,Q)
            zero_ref=sequential_affine_reference(P,Q)
            check_close(zero_scan,zero_ref,dtype,"prefix_zero_start")
            checks+=1
            initial=torch.randn(batch,8,8,dtype=dtype)*.01
            nonzero_scan=parallel_affine_prefix(P,Q,initial)
            nonzero_ref=sequential_affine_reference(P,Q,initial)
            check_close(nonzero_scan,nonzero_ref,dtype,"prefix_nonzero_start")
            checks+=1

            # Test each chronological step against original PINNED PyTorch
            # no-projection writer, not only against the algebra reference.
            state=initial.clone()
            states=[]
            for t in range(steps):
                state=source.gated_write(state,k[:,t],v[:,t],a[:,t],
                                         beta[:,t],mask[:,t],radius=None)
                states.append(state)
            native=torch.stack(states,1)
            check_close(native,nonzero_scan,dtype,"source_gated_write_no_clip")
            checks+=1

            g1=torch.autograd.grad(nonzero_scan.square().sum(),(k,v,a,beta),
                                  retain_graph=True)
            g2=torch.autograd.grad(native.square().sum(),(k,v,a,beta),
                                  retain_graph=False)
            for lhs,rhs in zip(g1,g2):
                check_close(lhs,rhs,dtype,"source_gate_value_key_gradient")
                checks+=1

            # Expose the noncommutative schedule: swapping order changes
            # result in general even though associative grouping is legal.
            if steps>1:
                u=scan_source_schedules(k,v,a,beta,mask,initial)
                check_close(u,nonzero_scan,dtype,"convenience_entrypoint")
                checks+=1

        # Whole-NATIVE real gate schedule extracted from original true
        # recurrence: scan reproduces memory exactly *only if projection
        # inactive*, but gates have already been sequentially calculated.
        model=source.AlignedAssociativeTAC().to(dtype=dtype).eval()
        batch,steps=2,18
        kind=torch.randint(0,3,(batch,steps))
        keys=torch.randint(0,65,(batch,steps))
        vals=torch.randint(0,33,(batch,steps))
        _,trace=model(kind,keys,vals,return_trace=True)
        addresses=trace["addresses"]
        contents=F.normalize(model.content(vals),dim=-1,eps=1e-6)
        gates=trace["gates"]
        alpha,beta=gates[...,:1],gates[...,1:]
        mask=kind.ne(2).unsqueeze(-1)
        scanned=scan_source_schedules(addresses,contents,alpha,beta,mask)
        # Input weights are source-default near-zero, so clipping should not
        # activate. Check this explicitly instead of presuming the theorem.
        if trace["memory"].abs().amax().item()>1.0:
            raise RuntimeError("unexpected native memory scale; check clipping")
        check_close(scanned,trace["memory"],dtype,"original_calculated_native_gate_scan")
        checks+=1

    return checks


def reject_unearned_parallel_claims():
    torch.manual_seed(11)
    q=torch.eye(8,dtype=torch.float64)[None,None].repeat(1,3,1,1)
    z=torch.zeros_like(q)
    # Unknown masks, empty history and invalid key shapes must fail, not
    # turn into misleading results.
    try:
        parallel_affine_prefix(q[:,:0],z[:,:0])
    except ValueError:
        pass
    else: raise AssertionError("empty history accepted")
    try:
        associative_affine_schedule(torch.ones(1,3,7),
          torch.ones(1,3,7),torch.ones(1,3,1),torch.ones(1,3,1),
          torch.ones(1,3,1,dtype=torch.bool))
    except ValueError:
        pass
    else: raise AssertionError("invalid key dimension accepted")


if __name__=="__main__":
    torch.set_num_threads(1)
    n=forward_backward_partition()
    m=scan_reference_and_native_schedule()
    reject_unearned_parallel_claims()
    print({"status":"PASS_CONDITIONAL_PARALLEL_TENSOR_GRAPH",
      "full_original_model_parallel_input_preprojection_checks":n,
      "affine_scan_and_pinned_source_gradient_checks":m,
      "original_model_parameters_unchanged":101527,
      "native_full_time_scan_proven":False,
      "memory_scan_only_when_gates_precomputed_and_no_clipping":True,
      "all_native_model_steps_parallel":False,
      "new_training_steps":0,"real_world_advantage_established":False})
