#!/usr/bin/env python3
"""No-fit actual-source causality checks; not predictive score or trained gain."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'research/mathematical-engineering/context026'),str(ROOT/'research/mathematical-engineering/langbase024')]
from context_transition import ContextTransition026
from byte_next_reference import original_source


def audit():
    torch.set_num_threads(2);torch.manual_seed(26026)
    fit,_,supp=original_source(ROOT);fit={**fit,supp[0]:supp[1]}
    model=ContextTransition026().eval();cases=[]
    with torch.no_grad():
        for name,raw in sorted(fit.items()):
            x=torch.tensor([list(raw[:64])],dtype=torch.long)
            future=x.clone();future[:,32:]=future[:,32:]^255
            y,t=model(x,return_trace=True);p,q=model(future,return_trace=True)
            if not torch.equal(t['native_query_key'][:,:32],q['native_query_key'][:,:32]):raise ValueError('actual FIT source key prefix changed')
            if not torch.allclose(y[:,:32],p[:,:32],atol=1e-6,rtol=0):raise ValueError('actual FIT prediction prefix changed')
            if not torch.equal(t['native_write_key'][:,1:],t['native_query_key'][:,:-1]):raise ValueError('actual FIT write timing differs')
            native_norm=float(torch.linalg.matrix_norm(t['native_memory'],ord='fro',dim=(-2,-1)).max())
            if native_norm>200*8**.5:raise ValueError('actual native norm exceeds registered loose bound')
            cases.append(dict(owner=name,source_sha256=hashlib.sha256(raw).hexdigest(),window_start=0,window_bytes=64,
                future_change_from=32,original_prefix_identity=True,shifted_write_identity=True,
                observed_native_norm_max=native_norm,status='PASS_SOURCE_CAUSALITY_UNTRAINED_CANDIDATE'))
    return dict(study='CONTEXT026',cases=cases,parameters=model.parameter_count(),trained=False,
        candidate_DEV_NLL='NOT_RUN',source_evidence='AUTHENTIC_FIT_WITH_SYNTHETIC_FUTURE_CONTROLS',
        full_IEEE_AD='OPEN',owner_carried_state='NOT_IMPLEMENTED_WINDOW_RESET_ONLY',protected_TEST=False)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(audit(),indent=2,sort_keys=True,allow_nan=False)+'\n')
