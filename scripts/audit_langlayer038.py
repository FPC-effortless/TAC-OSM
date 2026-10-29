#!/usr/bin/env python3
"""No-fit source/serial readiness. No DEV NLL, trained skill or qualification."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import urllib.request
import torch

ROOT=Path(__file__).resolve().parents[1]
STUDY=ROOT/'research/mathematical-engineering/langlayer038'
sys.path[:0]=[str(ROOT/'src'),str(STUDY),str(ROOT/'research/mathematical-engineering/langbase024')]
from recurrence_only import RecurrenceOnlyDepth038,RecurrenceOnlyUpperBlock
from byte_next_reference import original_source
from probe_adapter import UPPER_PROBE
from tac_osm.research_probe import probe_memory


def reference():
    link=json.loads((STUDY/'PNDS_MATH_LINK.json').read_text())
    for path,key in [(STUDY/'recurrence_only.py','source_sha256'),(ROOT/'research/mathematical-engineering/context026/context_transition.py','parent_sha256')]:
        if hashlib.sha256(path.read_bytes()).hexdigest()!=link[key]: raise ValueError('actual model source differs')
    base=f"https://raw.githubusercontent.com/FPC-effortless/pnds-lean/{link['pnds_commit']}/"
    contents={}
    for path,key in [(link['python_path'],'python_sha256'),(link['lean_path'],'lean_sha256'),(link['record_path'],'record_sha256')]:
        with urllib.request.urlopen(base+path,timeout=30) as response: raw=response.read()
        if hashlib.sha256(raw).hexdigest()!=link[key]: raise ValueError('pinned mathematical source differs')
        contents[path]=raw
    record=json.loads(contents[link['record_path']])
    if record['tac_commit']!=link['tac_source_commit'] or record['tac_source_sha256']!=link['source_sha256']: raise ValueError('reciprocal source differs')
    with tempfile.TemporaryDirectory() as tmp:
        p=Path(tmp)/'reference.py';p.write_bytes(contents[link['python_path']])
        spec=importlib.util.spec_from_file_location('independent038',p);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def error(a,b,tolerance):
    if a.shape!=b.shape or not torch.isfinite(a).all() or not torch.isfinite(b).all(): raise ValueError('invalid numeric correspondence')
    delta=(a-b).abs();scale=tolerance['atol']+tolerance['rtol']*torch.maximum(a.abs(),b.abs())
    return {'max_abs':float(delta.max().detach()),'scaled_error':float((delta/scale).max().detach())}


def compare(block,ref,z,tolerance):
    x=z.detach().clone().requires_grad_(True);rx=z.detach().clone().requires_grad_(True)
    y,tr=block(x,return_trace=True)
    weights={n:p.detach().clone().requires_grad_(True) for n,p in block.named_parameters()}
    ry,rtr=ref.serial_upper(weights,rx)
    if set(tr)!=set(rtr): raise ValueError('trace schema differs')
    forward={'output':error(y,ry,tolerance),**{n:error(v,rtr[n],tolerance) for n,v in tr.items()}}
    coeff=torch.sin(torch.arange(y.numel(),dtype=y.dtype)+1).reshape_as(y)
    gs=torch.autograd.grad((y*coeff).mean(),[*block.parameters(),x],allow_unused=True)
    gr=torch.autograd.grad((ry*coeff).mean(),[*weights.values(),rx],allow_unused=True)
    backward={};zeros=[]
    for (name,p),a,b in zip([*block.named_parameters(),('observed_features',x)],gs,gr):
        a=torch.zeros_like(p) if a is None else a;b=torch.zeros_like(p) if b is None else b
        backward[name]=error(a,b,tolerance)
        if not a.any() and not b.any(): zeros.append(name)
    failures=[n for n,v in {**forward,**{'VJP:'+n:v for n,v in backward.items()}}.items() if v['scaled_error']>1]
    return {'status':'FAIL' if failures else 'PASS_BOUNDED_UPPER_OPERATOR_ONLY','forward':forward,'vjps':backward,'zero_vjps':zeros,'failures':failures}


def audit():
    ref=reference();protocol=json.loads((STUDY/'PROTOCOL.json').read_text())
    torch.set_num_threads(2);fit,_,supp=original_source(ROOT);fit={**fit,supp[0]:supp[1]}
    cases=[];source=[];upper_probe=None
    for precision,tolerance in protocol['precisions'].items():
        torch.manual_seed(protocol['engineering_seed']);dtype=getattr(torch,precision)
        m=RecurrenceOnlyDepth038().to(dtype).eval();block=m.layernorm[1]
        if m.parameter_count()!=499901 or sum(p.numel() for p in m.parameters())!=499901: raise ValueError('actual parameter budget differs')
        for length in protocol['constructed_lengths']:
            z=torch.randn(2,length,280,dtype=dtype)
            cases.append(dict(kind='CONSTRUCTED_SOFTWARE_OPERATOR_CHECK',precision=precision,length=length,**compare(block,ref,z,tolerance)))
        for owner,raw in sorted(fit.items()):
            x=torch.tensor([list(raw[:33])],dtype=torch.long);future=x.clone();future[:,17:]^=255
            captured=[];handle=m.layernorm[0].register_forward_hook(lambda module,args,out:captured.append(out.detach()))
            try:
                with torch.no_grad(): a=m(x);b=m(future)
            finally: handle.remove()
            prefix=error(a[:,:17],b[:,:17],tolerance)
            r=compare(block,ref,captured[0],tolerance)
            cases.append(dict(kind='AUTHENTIC_FIT_UNTRAINED_UPPER_GRAPH_CHECK',owner=owner,source_sha256=hashlib.sha256(raw).hexdigest(),precision=precision,length=33,**r))
            source.append(dict(owner=owner,sha256=hashlib.sha256(raw).hexdigest(),precision=precision,prefix_error=prefix))
            if upper_probe is None:
                features=captured[0]
                sample={'observed_features':features[:,:-1],'query_features':features[:,-1]}
                upper_probe=probe_memory(block,sample,UPPER_PROBE)
    fail=any(c['failures'] for c in cases) or any(c['prefix_error']['scaled_error']>1 for c in source)
    return {'study':'LANGLAYER038','status':'FAIL' if fail else 'PASS_NO_FIT_SOURCE_AND_UPPER_CORRESPONDENCE',
        'architecture':m.architecture_record(),'cases':cases,'source_prefix':source,'upper_only_probe':upper_probe,
        'actual_FIT_owners':len(fit),'optimizer_steps':0,'candidate_DEV_NLL':'NOT_RUN','long_context_trained_utility':'NOT_RUN',
        'originals_035_036_037':'UNCHANGED','full_parent_memory_adapter':'INCONCLUSIVE_NOT_INSTRUMENTED',
        'full_composed_Lean_Python_IEEE_AD':'OPEN','matched_training_compute':'NOT_ESTABLISHED',
        'evidence_level':'internally_measured_not_independently_qualified','shared_dependency':'PyTorch primitives/autograd; one VJP scalar, finite cases only',
        'protected_TEST':False,'GPU_minutes':0}


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    try: result=audit()
    except Exception as e:
        a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps({'status':'BLOCKED_ERROR','reason':str(e)},indent=2)+'\n');raise
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+'\n')
    print(json.dumps({'status':result['status'],'cases':len(result['cases']),'parameters':result['architecture']['trainable_parameters'],'trained':False}))
    if result['status']=='FAIL': raise SystemExit(1)

if __name__=='__main__': main()
