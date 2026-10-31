#!/usr/bin/env python3
"""Bounded no-fit CPU profile on original exposed DEV, partial FLOPs only."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from diagnose_adaptive_byte_token_nofit import SOURCE_BLOBS,git_blob,SOURCE_COMMIT
from tac_osm.adaptive_token_controls import ARM_NAMES,TokenComparisonArm


def profile(root):
    import torch
    torch.set_num_threads(1)
    files=[]
    raw_sources=[]
    # Verify all six original files, use four fixed exposed 256-byte prefixes.
    for path,expected in SOURCE_BLOBS.items():
        raw=(root/path).read_bytes()
        if git_blob(raw)!=expected:
            raise ValueError('publisher original mismatch: '+path)
        files.append({'path':path,'sha256':hashlib.sha256(raw).hexdigest()})
        raw_sources.append(raw[:256])
    targets=torch.tensor([list(raw) for raw in raw_sources[:4]],dtype=torch.long)
    ids=torch.cat((torch.full((4,1),256,dtype=torch.long),targets[:,:-1]),1)
    draws=torch.rand((4,255),generator=torch.Generator().manual_seed(20261031))
    rows=[]
    for name in ARM_NAMES:
        model=TokenComparisonArm(name,seed=1,random_byte_probability=.5 if name=='random_gate' else None)
        start=time.perf_counter()
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU],with_flops=True) as instrument:
            logits=model(ids,mode='surrogate',random_draws=draws if name=='random_gate' else None)
            loss=torch.nn.functional.cross_entropy(logits.reshape(-1,256),targets.flatten())
            loss.backward()  # diagnostic gradient only, never optimizer construction
        elapsed=time.perf_counter()-start
        operations=instrument.key_averages()
        reported=sum(e.flops or 0 for e in operations)
        missing=[{'operator':e.key,'calls':e.count} for e in operations if not e.flops]
        gradients={}
        for module,layer in model.backbone.named_children():
            ps=list(layer.parameters())
            gradients[module]={'trainable':sum(p.numel() for p in ps if p.requires_grad),
                               'nonzero_gradient_parameters':sum(p.numel() for p in ps if p.grad is not None and bool(p.grad.abs().sum()>0)),
                               'all_recorded_gradients_finite':all(bool(torch.isfinite(p.grad).all()) for p in ps if p.grad is not None)}
        rows.append({'arm':name,'reported_operator_flops':reported,'operators_without_flop_counts':missing,
                     'instrumented_wall_seconds':elapsed,'gradient_modules':gradients,
                     'full_compute_coverage':'INCONCLUSIVE','parameter_receipt':model.parameter_receipt()})
    counts=[r['reported_operator_flops'] for r in rows]
    return {'study':'SENSE-TOK-006','source_commit':SOURCE_COMMIT,'sources':files,'scope':'EXPOSED_DEV_NO_FIT_PARTIAL_CPU_PROFILE',
            'batch':4,'history_bytes':256,'seed':1,'torch_version':torch.__version__,'platform':platform.platform(),
            'reported_operator_flop_relative_spread':(max(counts)-min(counts))/max(counts),
            'rows':rows,'optimizer_steps':0,'fits':0,'test_read':False,
            'full_training_compute_parity':'INCONCLUSIVE','training_authorized':False,
            'limitations':['Profiler omits FLOPs for listed operators; these are not full training FLOPs.',
                           'Only one untrained seed and one exposed source batch; not full training behavior.',
                           'Wall time includes profiler overhead and is not a fairness requirement.',
                           'Random probability .5 is a diagnostic fixture, not FIT calibration.']}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    a.output.write_text(json.dumps(profile(a.source_root),indent=2)+'\n')
