#!/usr/bin/env python3
"""Partial source operator parity; full trained graph/Float32/AD still OPEN."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'research/mathematical-engineering/langmem084'))
from operators import byte_book,update_matrix

def main():
    p=argparse.ArgumentParser();p.add_argument('--reference',type=Path,required=True);a=p.parse_args()
    spec=importlib.util.spec_from_file_location('pnds084',a.reference);ref=importlib.util.module_from_spec(spec);spec.loader.exec_module(ref)
    expected=torch.tensor([ref.byte_code(b) for b in range(256)],dtype=torch.float64)
    assert torch.allclose(expected,byte_book(torch.float64),atol=1e-15,rtol=1e-15)
    g=torch.Generator().manual_seed(84);max_error=0.
    for _ in range(32):
        M=torch.randn(8,8,dtype=torch.float64,generator=g)
        k=torch.nn.functional.normalize(torch.randn(8,dtype=torch.float64,generator=g),dim=0)
        v=torch.randn(8,dtype=torch.float64,generator=g)
        for mode in ('source','overwrite'):
            result=update_matrix(M,k,v,mode=mode,rho=.99,beta=.02,epsilon=.005)
            if mode=='source':r=ref.source_update(M.tolist(),k.tolist(),v.tolist(),.99,.02,.005)
            else:r=ref.overwrite(M.tolist(),k.tolist(),v.tolist(),.99)
            independent=torch.tensor(r,dtype=torch.float64)
            error=float((result-independent).abs().max());max_error=max(error,max_error)
            if error>1e-12:raise ValueError('independent PNDS operator mismatch')
    print(json.dumps({'status':'PASS_PARTIAL_OPERATOR_PARITY','max_float64_error':max_error,'byte_codes':256,'randomized_operator_pairs':64,'whole_source_IEEE_AD':'OPEN'}))
if __name__=='__main__':main()
