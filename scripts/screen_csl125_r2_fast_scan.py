"""Same-weights no-fit source graph timing/parity of UNPROMOTED fast scan.

Temporary Python binding of the module operator is controlled only inside
this diagnostic. Neither trained model nor production CSL125R2 source uses
that replacement. Original operator is restored in finally.
"""
from __future__ import annotations
import hashlib,json,statistics,time
from pathlib import Path
import torch
import tac_osm.csl125_r2 as source
from tac_osm.csl125_r2_fast_scan import bounded_cumprod_scan
ROOT=Path(__file__).resolve().parents[1]

def take():
    torch.set_num_threads(1);torch.manual_seed(125300)
    model=source.CSL125R2().eval()
    x=torch.randint(0,32768,(1,64))
    original=source.selective_affine_scan
    outcomes={}
    try:
        with torch.inference_mode():
            for key,fn in [('stable_affine',original),('bounded_cumprod_exploratory',bounded_cumprod_scan)]:
                source.selective_affine_scan=fn
                for _ in range(2):model(x,depth=20,stream_ids=('source',),all_rungs=True,emit_token_logits=False)
                samples=[]
                for _ in range(4):
                    start=time.perf_counter()
                    outs,state=model(x,depth=20,stream_ids=('source',),all_rungs=True,emit_token_logits=False)
                    samples.append((time.perf_counter()-start)*1000)
                outcomes[key]={'ms':samples,'median_ms':statistics.median(samples),
                    'outputs':{r:{k:v.detach().clone() for k,v in values.items()} for r,values in outs.items()},
                    'layer_state':[(s.fast.clone(),s.slow.clone(),s.fast_mass.clone(),s.slow_mass.clone()) for s in state.layers]}
    finally:
        source.selective_affine_scan=original
    base=outcomes['stable_affine'];fast=outcomes['bounded_cumprod_exploratory']
    maxlogit=max(float((base['outputs'][r][k]-fast['outputs'][r][k]).abs().max())
                 for r in (2,8,20) for k in base['outputs'][r])
    maxstate=max(float((x-y).abs().max()) for a,b in zip(base['layer_state'],fast['layer_state']) for x,y in zip(a,b))
    receipt={'stage':'NO_FIT_SAME_WEIGHTS_EXPLORATORY_KERNEL_SCREEN','model_source_sha256':hashlib.sha256((ROOT/'src/tac_osm/csl125_r2.py').read_bytes()).hexdigest(),
        'candidate_source_sha256':hashlib.sha256((ROOT/'src/tac_osm/csl125_r2_fast_scan.py').read_bytes()).hexdigest(),
        'parameters':model.parameter_count(),'device':'cpu','batch':1,'seq':64,'depth':20,'rungs':[2,8,20],
        'emitted_token_logits':False,'optimizer_steps':0,'repeats':4,
        'stable_ms':base['ms'],'fast_ms':fast['ms'],'stable_median_ms':base['median_ms'],
        'fast_median_ms':fast['median_ms'],'max_output_logit_difference':maxlogit,
        'max_state_difference':maxstate,'candidate_is_in_production_model':False,
        'limits':'CPU FP32 one thread; no future input, no training, no GPU AMP or fused-kernel equivalence; does not justify promotion.'}
    assert maxlogit<5e-4 and maxstate<1e-5
    p=ROOT/'research/mathematical-engineering/csl125_r2/FAST_SCAN_SCREEN_NO_FIT.json'
    p.write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps({k:receipt[k] for k in ('stable_median_ms','fast_median_ms','max_output_logit_difference','max_state_difference')},indent=2))
    return receipt

if __name__=='__main__':take()
