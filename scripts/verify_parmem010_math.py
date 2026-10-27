#!/usr/bin/env python3
"""No-fit cross-repository mathematical correspondence receipt.

Requires an explicit PNDS checkout; fails rather than skipping missing refs.
No optimizer/data loader/remote job, no protected FIT/DEV/TEST evaluation.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from tac_osm.variants.source_aligned_parallel_world import SourceAlignedParallelWorld
from tac_osm.variants.parallel_affine_memory import associative_affine_schedule

SOURCE_SHA="63ea0aa67904fa946b9baa69d2e4b0cc23dc1c9a"
SOURCE_FILES=["source_aligned_parallel_world.py","parallel_time_age_world.py",
              "parallel_residual_world.py","parallel_continuous_world.py",
              "parallel_affine_memory.py"]


def load_reference(root):
    path=root/"foundation/v1/source_aligned_model_math.py"
    spec=importlib.util.spec_from_file_location("pnds_parmem010_reference",path)
    module=importlib.util.module_from_spec(spec)
    sys.modules[spec.name]=module
    spec.loader.exec_module(module)
    assert module.SOURCE_COMMIT==SOURCE_SHA
    return module,path


def as_numpy(x):
    return x.detach().cpu().numpy().copy()


def weights(model):
    return {k:as_numpy(v) for k,v in model.state_dict().items()}


def source_pins():
    result={}
    for name in SOURCE_FILES:
        path="src/tac_osm/variants/"+name
        original=subprocess.check_output(["git","show",SOURCE_SHA+":"+path],cwd=ROOT)
        current=(ROOT/path).read_bytes()
        if current != original:
            raise AssertionError("source changed from pinned implementation: "+path)
        result[path]=hashlib.sha256(current).hexdigest()
    return result


class Checks:
    def __init__(self):
        self.results=[]

    def close(self,name,actual,expected,atol,rtol):
        a=as_numpy(actual) if torch.is_tensor(actual) else np.asarray(actual)
        b=as_numpy(expected) if torch.is_tensor(expected) else np.asarray(expected)
        assert a.shape==b.shape,(name,a.shape,b.shape)
        assert np.isfinite(a).all() and np.isfinite(b).all(),name
        maximum=float(np.max(np.abs(a-b),initial=0.))
        np.testing.assert_allclose(a,b,atol=atol,rtol=rtol,err_msg=name)
        self.results.append(dict(check=name,max_absolute_error=maximum,
                                 max_actual_magnitude=float(np.max(np.abs(a),initial=0.)),
                                 max_reference_magnitude=float(np.max(np.abs(b),initial=0.)),
                                 atol=atol,rtol=rtol))


def model_fixture(seed,dtype):
    torch.manual_seed(seed)
    model=SourceAlignedParallelWorld().to(dtype=dtype)
    # Exercise paths hidden by zero-initialized temporal/predictive weights.
    with torch.no_grad():
        model.predictive_memory_delta.weight.normal_(0,.1)
        model.backbone.core.time_projection.weight.normal_(0,.03)
        model.retention.weight.normal_(0,.02)
        model.write_strength.weight.normal_(0,.02)
    return model


def inputs(t,dtype,pattern="mixed"):
    x=torch.randn(2,t,14,dtype=dtype)*.2
    action=torch.randn_like(x)*.3
    dt=torch.rand(2,t,dtype=dtype)+.1
    if pattern=="none": mask=torch.zeros(2,t,dtype=torch.bool)
    elif pattern=="all": mask=torch.ones(2,t,dtype=torch.bool)
    else: mask=torch.arange(t)[None,:].expand(2,t)%3!=1
    owners=torch.tensor([31,57],dtype=torch.int64)
    events=torch.arange(t,dtype=torch.int64)[None,:].expand(2,t)*3
    events=events+torch.tensor([[0],[2**40+37]],dtype=torch.int64)
    return x,action,dt,mask,owners,events


def batch(model,data):
    x,a,dt,mask,owners,events=data
    return model(x,a,elapsed_dt=dt,write_mask=mask,source_ids=owners,
                 event_index=events,return_trace=True)


def stream(model,data):
    x,a,dt,mask,owners,events=data
    state=model.init_stream(owners)
    predictions=[]
    for t in range(x.shape[1]):
        y,state,trace=model.stream_step(state,x[:,t],a[:,t],elapsed_dt=dt[:,t],
                                       write_mask=mask[:,t],source_ids=owners,
                                       event_index=events[:,t],return_trace=True)
        predictions.append(y)
    return torch.stack(predictions,1),state


def forward_checks(ref,checks):
    for dtype in (torch.float32,torch.float64):
        for t,pattern in [(1,"none"),(2,"all"),(3,"mixed"),(7,"mixed"),
                          (16,"none"),(33,"all"),(129,"mixed")]:
            model=model_fixture(9010+t,dtype)
            data=inputs(t,dtype,pattern)
            independent=ref.SourceAlignedModelReference(weights(model))
            with torch.no_grad():
                y,trace=batch(model,data)
                ys,st=stream(model,data)
                yn,sn,tn=independent.forward(*(as_numpy(v) for v in data))
                label=f"{str(dtype).split('.')[-1]}/T{t}/{pattern}"
                atol,rtol=((2e-5,2e-4) if dtype==torch.float32 else (2e-12,2e-10))
                checks.close(label+"/numpy_output",y,yn,atol,rtol)
                checks.close(label+"/stream_output",y,ys,atol,rtol)
                for k in tn:
                    checks.close(label+"/numpy_trace/"+k,trace[k],tn[k],atol,rtol)
                for k in ("memory","aligned_memory"):
                    states=trace[k]
                    previous=torch.cat((torch.zeros_like(states[:,:1]),states[:,:-1]),1)
                    checks.close(label+"/bitwise_nonwrite/"+k,
                                 states[~data[3]],previous[~data[3]],0.,0.)
                checks.close(label+"/stream_memory",trace["aligned_memory"][:,-1],
                             st.aligned_memory,atol,rtol)
                for field,actual in [("fast_sum",st.native.fast_sum),("memory",st.native.memory),
                                     ("clock",st.native.elapsed_clock),
                                     ("last_write",st.native.last_write_clock),
                                     ("aligned",st.aligned_memory)]:
                    checks.close(label+"/numpy_state/"+field,actual,getattr(sn,field),atol,rtol)
                assert st.native.count==sn.count==t
                assert np.array_equal(as_numpy(st.native.write_seen),sn.seen)
                snapshot=st.aligned_memory.clone()
                requested=data[-1][:,0]
                historical=model.historic_query(st,source_ids=data[-2],requested_event_index=requested)
                checks.close(label+"/numpy_history",historical,
                             independent.query(sn,as_numpy(data[-2]),as_numpy(requested)),atol,rtol)
                assert torch.equal(snapshot,st.aligned_memory)
                # Same residues imply identical key AND read router and query output.
                low=torch.tensor([1,2],dtype=torch.int64)
                high=low+155701
                ka,pa=model._keys(low,dtype)
                kb,pb=model._keys(high,dtype)
                assert torch.equal(ka,kb) and torch.equal(pa,pb)
                assert torch.equal(model._read(st.aligned_memory,low)[0],
                                   model._read(st.aligned_memory,high)[0])


def vjp_checks(ref,checks):
    for seed in range(5):
        torch.manual_seed(10010+seed)
        m=torch.randn(16,8,dtype=torch.float64,requires_grad=True)
        k=torch.randn(8,dtype=torch.float64,requires_grad=True)
        if seed==0:
            with torch.no_grad(): k.zero_()
        v=torch.randn(16,dtype=torch.float64,requires_grad=True)
        a=torch.tensor(.93,dtype=torch.float64,requires_grad=True)
        b=torch.tensor(.28,dtype=torch.float64,requires_grad=True)
        upstream=torch.randn(16,8,dtype=torch.float64)
        for admitted in (False,True):
            # Actual source kernel takes square value blocks; concatenate two.
            states=[]
            for block in range(2):
                p,q=associative_affine_schedule(k[None,None],v[block*8:block*8+8][None,None],
                                                a.reshape(1,1,1),b.reshape(1,1,1),
                                                torch.tensor([[[admitted]]]))
                states.append(m[block*8:block*8+8]@p[0,0]+q[0,0])
            grads=torch.autograd.grad((torch.cat(states)*upstream).sum(),(m,k,v,a,b))
            explicit=ref.write_vjp(*(as_numpy(z) for z in (m,k,v,a,b,upstream)),admitted)
            for name,g,e in zip(("memory","key","value","alpha","beta"),grads,explicit):
                checks.close(f"analytic_vjp/{seed}/{admitted}/{name}",g,e,2e-12,2e-10)


def gradient_checks(ref,checks):
    model=model_fixture(9110,torch.float64)
    data=list(inputs(5,torch.float64))
    for i in range(3): data[i].requires_grad_()
    x,a,dt,mask,owners,events=data
    y,trace=batch(model,data)
    ys,ss=stream(model,data)
    upstream=torch.randn_like(y)
    um=torch.randn_like(ss.aligned_memory)*.2
    uh=torch.randn(2,14,dtype=torch.float64)*.2
    hist=lambda m:model.historical_content(model._read(m,events[:,0])[0])
    objective=lambda y,m,h:(y*upstream).sum()+(m*um).sum()+(h*uh).sum()
    lb=objective(y,trace["aligned_memory"][:,-1],hist(trace["aligned_memory"][:,-1]))
    ls=objective(ys,ss.aligned_memory,hist(ss.aligned_memory))
    parameters=list(model.named_parameters())
    tensors=[x,a,dt]+[p for _,p in parameters]
    names=["observation","action","elapsed_dt"]+[n for n,_ in parameters]
    gb=torch.autograd.grad(lb,tensors)
    gs=torch.autograd.grad(ls,tensors)
    for name,b,s in zip(names,gb,gs):
        assert torch.count_nonzero(b).item()>0,"unexercised gradient group: "+name
        checks.close("batch_stream_gradient/"+name,b,s,1e-10,1e-7)
    baseline=weights(model)
    arrays=[as_numpy(v) for v in data]
    # Independent centered finite differences for EVERY learned tensor and
    # each floating input group. One seeded direction/group; no fit occurs.
    rng=np.random.default_rng(9110)
    h=2e-5
    def loss(w,values):
        reference=ref.SourceAlignedModelReference(w)
        yn,sn,_=reference.forward(*values)
        hn=reference.query(sn,values[-2],values[-1][:,0])
        return float((yn*as_numpy(upstream)).sum()+(sn.aligned*as_numpy(um)).sum()+
                     (hn*as_numpy(uh)).sum())
    checks.close("independent_loss",lb,loss(baseline,arrays),2e-12,2e-10)
    for index,(name,g,tensor) in enumerate(zip(names,gb,tensors)):
        direction=rng.normal(size=tuple(tensor.shape))
        direction=direction/max(float(np.linalg.norm(direction)),1.)
        wp={k:v.copy() for k,v in baseline.items()}
        wm={k:v.copy() for k,v in baseline.items()}
        ap=[v.copy() for v in arrays]
        am=[v.copy() for v in arrays]
        if index<3:
            ap[index]+=h*direction
            am[index]-=h*direction
        else:
            wp[name]+=h*direction
            wm[name]-=h*direction
        numeric=(loss(wp,ap)-loss(wm,am))/(2*h)
        automatic=float(np.sum(as_numpy(g)*direction))
        checks.close("independent_directional_derivative/"+name,automatic,numeric,5e-8,5e-5)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pnds-root",required=True,type=Path)
    parser.add_argument("--output",required=True,type=Path)
    args=parser.parse_args()
    torch.set_num_threads(1)
    ref,path=load_reference(args.pnds_root.resolve())
    pins=source_pins()
    checks=Checks()
    forward_checks(ref,checks)
    vjp_checks(ref,checks)
    gradient_checks(ref,checks)
    receipt=dict(schema="parmem010.math_correspondence.v1",source_commit=SOURCE_SHA,
                 source_sha256=pins,reference_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                 torch_version=torch.__version__,numpy_version=np.__version__,
                 device="cpu",optimizer_steps=0,real_data_evaluation=False,
                 status="PASS_NUMERICAL_FIXTURES_NOT_FORMAL_REFINEMENT",
                 checks=checks.results)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(receipt,indent=2)+"\n")
    print(json.dumps(dict(status=receipt["status"],checks=len(checks.results),
                          output=str(args.output)),sort_keys=True))


if __name__=="__main__":
    main()
