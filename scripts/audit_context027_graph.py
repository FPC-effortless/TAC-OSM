#!/usr/bin/env python3
"""No-fit full serial/scan correspondence; no predictive score or qualification."""
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
STUDY=ROOT/'research/mathematical-engineering/context027'
sys.path[:0]=[str(ROOT/'research/mathematical-engineering/context026'),str(ROOT/'research/mathematical-engineering/langbase024')]
from context_transition import ContextTransition026
from byte_next_reference import original_source


def load_reference(path=None):
    if path is None:
        link=json.loads((STUDY/'PNDS_GRAPH_LINK.json').read_text())
        url=f"https://raw.githubusercontent.com/FPC-effortless/pnds-lean/{link['pnds_commit']}/{link['python_path']}"
        with urllib.request.urlopen(url,timeout=30) as response: raw=response.read()
        if hashlib.sha256(raw).hexdigest()!=link['python_sha256']: raise ValueError('PNDS graph reference hash differs')
        base=f"https://raw.githubusercontent.com/FPC-effortless/pnds-lean/{link['pnds_commit']}/"
        for path,key in [(link['record_path'],'record_sha256'),(link['lean_path'],'lean_sha256')]:
            with urllib.request.urlopen(base+path,timeout=30) as response: content=response.read()
            if hashlib.sha256(content).hexdigest()!=link[key]: raise ValueError('paired graph formulation differs')
            if key=='record_sha256':
                record=json.loads(content)
                if record['tac_source_sha256']!=link['tac_source_sha256'] or record['tac_commit']!=link['tac_commit']:
                    raise ValueError('reciprocal candidate source differs')
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'graph.py';p.write_bytes(raw)
            return load_reference(p)
    spec=importlib.util.spec_from_file_location('independent_context_graph',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def comparison(a,b,atol,rtol):
    if a.shape!=b.shape: raise ValueError(f'shape mismatch {a.shape}/{b.shape}')
    if a.dtype==torch.bool or b.dtype==torch.bool:
        return {'max_abs':0. if torch.equal(a,b) else 1.,'scaled_error':0. if torch.equal(a,b) else 2.}
    if not torch.isfinite(a).all() or not torch.isfinite(b).all(): raise ValueError('nonfinite correspondence')
    d=(a-b).abs(); scale=atol+rtol*torch.maximum(a.abs(),b.abs())
    return {'max_abs':float(d.max().detach()),'scaled_error':float((d/scale).max().detach())}


def one_case(model,reference,x,mask,atol,rtol):
    y,tr=model(x,write_mask=mask,return_trace=True)
    weights={n:p.detach().clone().requires_grad_(True) for n,p in model.named_parameters()}
    ry,rtr=reference.serial_graph(weights,x,mask)
    if set(tr)!=set(rtr): raise ValueError('trace schema mismatch')
    forward={'logits':comparison(y,ry,atol,rtol)}
    forward.update({name:comparison(value,rtr[name],atol,rtol) for name,value in tr.items()})
    # Deterministic label-blind VJP, not a training loss or an optimizer step.
    coeff=torch.sin(torch.arange(y.numel(),device=y.device,dtype=y.dtype)+1).reshape_as(y)
    actual=torch.autograd.grad((y*coeff).mean(),tuple(model.parameters()),allow_unused=True)
    expected=torch.autograd.grad((ry*coeff).mean(),tuple(weights.values()),allow_unused=True)
    backward={}; zero=[]; disconnected=[]
    for (name,p),g,h in zip(model.named_parameters(),actual,expected):
        if g is None or h is None: disconnected.append({'parameter':name,'source':g is None,'reference':h is None})
        g=torch.zeros_like(p) if g is None else g; h=torch.zeros_like(p) if h is None else h
        backward[name]=comparison(g,h,atol,rtol)
        if not g.any() and not h.any(): zero.append(name)
    failed=[name for name,error in {**forward,**{'grad:'+n:e for n,e in backward.items()}}.items() if error['scaled_error']>1]
    return {'status':'FAIL' if failed else 'PASS_NUMERICAL_CORRESPONDENCE_ONLY','failed':failed,
        'forward':forward,'parameter_vjps':backward,'zero_vjps':zero,'disconnected_autograd_leaves':disconnected}


def audit(reference):
    protocol=json.loads((STUDY/'PROTOCOL.json').read_text())
    source=(ROOT/protocol['source_path']).read_bytes()
    if hashlib.sha256(source).hexdigest()!=protocol['source_sha256']: raise ValueError('actual candidate source differs')
    torch.set_num_threads(2);torch.manual_seed(protocol['seeds'][0])
    fit,_,supp=original_source(ROOT);fit={**fit,supp[0]:supp[1]}
    results=[];controls=[]
    for precision,tolerance in protocol['precisions'].items():
        dtype=getattr(torch,precision);torch.manual_seed(protocol['seeds'][0]);m=ContextTransition026().to(dtype).eval()
        count=m.parameter_count()
        if count!=500122 or sum(p.numel() for p in m.parameters())!=count: raise ValueError('untrained model size differs')
        before={n:p.detach().clone() for n,p in m.named_parameters()}
        for length in protocol['fixture_lengths']:
            x=torch.randint(0,256,(protocol['fixture_batch'],length))
            for name in protocol['masks']:
                mask=torch.ones_like(x,dtype=torch.bool) if name=='all' else torch.zeros_like(x,dtype=torch.bool) if name=='none' else (torch.arange(length)[None,:]+torch.arange(x.size(0))[:,None])%2==0
                result=one_case(m,reference,x,mask,**tolerance)
                results.append(dict(kind='CONSTRUCTED_SOFTWARE_FIXTURE',precision=precision,length=length,mask=name,**result))
        for owner,raw in sorted(fit.items()):
            x=torch.tensor([list(raw[:17])],dtype=torch.long);mask=torch.ones_like(x,dtype=torch.bool)
            result=one_case(m,reference,x,mask,**tolerance)
            results.append(dict(kind='AUTHENTIC_FIT_UNTRAINED_GRAPH_CHECK',owner=owner,source_sha256=hashlib.sha256(raw).hexdigest(),precision=precision,length=17,mask='all',**result))
        # Prefix poison checks both full implementations; no future labels exist.
        owners=sorted(fit)[:2]
        x=torch.tensor([list(fit[owner][:17]) for owner in owners])
        changed=x.clone();changed[:,9:]^=255
        other_owner=x.clone();other_owner[1]^=255
        for implementation in ('source','reference'):
            with torch.no_grad():
                if implementation=='source': a=m(x);b=m(changed);single=m(x[:1]);isolated=m(other_owner)
                else:
                    w=dict(m.named_parameters());a,_=reference.serial_graph(w,x);b,_=reference.serial_graph(w,changed);single,_=reference.serial_graph(w,x[:1]);isolated,_=reference.serial_graph(w,other_owner)
            controls.append(dict(implementation=implementation,precision=precision,
                owners=owners,prefix=comparison(a[:,:9],b[:,:9],**tolerance),owner=comparison(a[:1],single,**tolerance),other_owner_poison=comparison(a[:1],isolated[:1],**tolerance)))
        if any(not torch.equal(before[n],p) for n,p in m.named_parameters()): raise ValueError('diagnostic changed model weights')
    failed=[i for i,r in enumerate(results) if r['failed']]
    bad_controls=[r for r in controls if max(r['prefix']['scaled_error'],r['owner']['scaled_error'],r['other_owner_poison']['scaled_error'])>1]
    return dict(study=protocol['study_id'],status='FAIL' if failed or bad_controls else 'PASS_FULL_FORWARD_AND_PARAMETER_VJP_NUMERICAL_CASES_ONLY',
        source_sha256=protocol['source_sha256'],cases=results,controls=controls,failed_cases=failed,
        learned_parameters=count,trained=False,optimizer_steps=0,DEV_NLL='NOT_RUN',protected_TEST=False,
        independent_qualification=False,evidence_level='internally_measured_not_independently_qualified',
        shared_dependency='PyTorch kernels and autograd; independent operator composition, not independent AD or IEEE proof',
        still_open=['full Lean input-to-logit graph/proofs','global IEEE error bounds','formal AD proof','trained memory utility','parameter/data/compute matched SSM','new fit authorization'])


def main():
    p=argparse.ArgumentParser();p.add_argument('--reference',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    try: result=audit(load_reference(a.reference))
    except Exception as error:
        result={'study':'CONTEXT027','status':'BLOCKED_ERROR','error_type':type(error).__name__,'reason':str(error),'trained':False,'optimizer_steps':0}
        a.output.parent.mkdir(parents=True,exist_ok=True)
        a.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
        raise
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+'\n')
    print(json.dumps({k:result[k] for k in ('status','failed_cases','learned_parameters','trained')}))
    if result['status']=='FAIL': raise SystemExit(1)

if __name__=='__main__': main()
