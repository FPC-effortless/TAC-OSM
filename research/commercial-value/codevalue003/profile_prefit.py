"""Genuine FIT B8x128 operator profile; no optimizer step, incomplete FLOPs."""
import argparse
import json
from pathlib import Path
import time

import torch
from core import ARMS,atomic_json,make,read_sources,sampler

def run(data,out):
    torch.set_num_threads(2)
    receipt,rows=read_sources(data,{'FIT'})
    docs={r['repo']+':'+r['path']:b for r,b in rows}
    _,batch,_=next(sampler(docs,1701,1))
    z=torch.tensor([list(b) for _,_,b in batch])
    report={'study':'CODEVALUE003','training_updates':0,'scope':'PREFIT_CPU_ORIGINAL_B8x128',
            'full_FLOP_coverage':False,'matched_FLOPs':False,
            'source_cohort_sha256':receipt['cohort_sha256'],'models':{}}
    for arm in ARMS:
        torch.manual_seed(1701);model=make(arm,'cpu').train()
        start=time.perf_counter()
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU],with_flops=True) as prof:
            logits=model(z[:,:-1])
            loss=torch.nn.functional.cross_entropy(logits.reshape(-1,256),z[:,1:].reshape(-1))
            loss.backward()
        events=prof.key_averages()
        report['models'][arm]={'trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad),
            'deployed_parameters':sum(p.numel() for p in model.parameters()),
            'supported_operator_FLOPs_forward_backward':sum(e.flops for e in events),
            'unaccounted_operator_names':sorted(e.key for e in events if e.flops==0),
            'wall_seconds':time.perf_counter()-start,'loss':float(loss.detach()),
            'caveat':'PyTorch profiler does not count every elementwise, solve, scan or backward operation; cannot establish full training compute parity.'}
    atomic_json(out,report)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);a=p.parse_args();run(a.data,a.out)
