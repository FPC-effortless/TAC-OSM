#!/usr/bin/env python3
"""Float64 operator parity against pinned PNDS reference; not full graph/AD."""
import argparse,importlib.util,json
from pathlib import Path
import torch
from scripts.langmem082_recover import read,RawRing

def main():
    p=argparse.ArgumentParser();p.add_argument('--reference',type=Path,required=True);a=p.parse_args()
    spec=importlib.util.spec_from_file_location('pnds082',a.reference);ref=importlib.util.module_from_spec(spec);spec.loader.exec_module(ref)
    g=torch.Generator().manual_seed(82);max_error=0.
    for _ in range(32):
        m=torch.randn(8,8,dtype=torch.float64,generator=g);q=torch.randn(8,dtype=torch.float64,generator=g)
        actual=read(m,q);expected=torch.tensor(ref.read(m.tolist(),q.tolist()),dtype=torch.float64)
        max_error=max(max_error,float((actual-expected).abs().max()))
        if not torch.allclose(actual,expected,atol=1e-12,rtol=1e-12):raise ValueError('matrix read reference mismatch')
        v=torch.randn(8,dtype=torch.float64,generator=g);wrong=torch.randn(8,dtype=torch.float64,generator=g)
        event=v[:,None]*q[None,:]
        actual_wrong=(wrong*torch.linalg.vector_norm(v)/torch.linalg.vector_norm(wrong))[:,None]*((v[None,:]@event)/(v*v).sum())
        expected_wrong=torch.tensor(ref.norm_matched_wrong_event(event.tolist(),v.tolist(),wrong.tolist()),dtype=torch.float64)
        if not torch.allclose(actual_wrong,expected_wrong,atol=1e-12,rtol=1e-12):raise ValueError('wrong event parity')
    for budget in (256,4160):
        ring=RawRing(budget)
        for t in range(5000):ring.append(t%256)
        cap=ref.capacity(budget)
        for a in range(5001):
            if (ring.get(a) is not None)!=ref.ring_available(5000,cap,a):raise ValueError('packed ring reference mismatch')
    print(json.dumps({'status':'PASS_PARTIAL_OPERATOR_PARITY','max_float64_read_error':max_error,'full_source_IEEE_AD':'OPEN'}))
if __name__=='__main__':main()
