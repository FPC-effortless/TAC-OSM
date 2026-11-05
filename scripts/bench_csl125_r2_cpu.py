"""Bounded no-fit CPU reference benchmark; not GPU/mobile/quality evidence."""
from __future__ import annotations
import hashlib,json,os,statistics,time
from pathlib import Path
import torch
from tac_osm.csl125_r1 import CSL125R1
from tac_osm.csl125_r2 import CSL125R2


def timed(fn,n=3):
    for _ in range(2):fn()
    vals=[]
    for _ in range(n):
        t=time.perf_counter();fn();vals.append((time.perf_counter()-t)*1000)
    return {'median_ms':round(statistics.median(vals),4),'samples_ms':[round(x,4) for x in vals]}


def state_bytes(state):
    total=0
    for layer in state.layers:
        for name in ('fast','slow','conv_tail','fast_mass','slow_mass'):
            total+=getattr(layer,name).numel()*getattr(layer,name).element_size()
    for cache in state.attention:
        if cache is not None:
            total += cache.k.numel()*cache.k.element_size()+cache.v.numel()*cache.v.element_size()
    return total


def run():
    torch.set_num_threads(1)
    out={'stage':'NO_FIT_REFERENCE_CPU','torch':torch.__version__,
         'threads':torch.get_num_threads(),'seed':1252,
         'batch':1,'prefill_tokens':64,'decode_tokens':8,'optimizer_steps':0,
         'full_vocabulary_projection':False,'device':'cpu','replicates_per_arm':3,
         'limitations':'No tokenizer, trained loss, CUDA, kernel fusion, energy or model utility. ' +
         'CPU PyTorch runtime includes Python/kernel overhead; no matched FLOPs claim.',
         'arms':{}}
    torch.manual_seed(1252)
    x=torch.randint(0,32768,(1,64))
    y=torch.randint(0,32768,(1,8))
    for name,cls in [('R1',CSL125R1),('R2',CSL125R2)]:
        torch.manual_seed(1252)
        m=cls().eval()
        row={'parameters':m.parameter_count(),'rungs':{}}
        with torch.inference_mode():
            for depth in (2,8,20):
                def prefill():
                    return m(x,depth=depth,emit_token_logits=False,stream_ids=('bench',))
                def decode():
                    st=m.initial_state(1,torch.device('cpu'),torch.float32,
                         depth=depth,stream_ids=('bench',))
                    for i in range(8):
                        _,st=m(y[:,i:i+1],depth=depth,state=st,
                               stream_ids=('bench',),emit_token_logits=False)
                    return st
                pre=timed(prefill)
                dec=timed(decode)
                st=prefill()[1]
                row['rungs'][str(depth)]={'prefill':pre,'decode':dec,
                    'prefill_tps':round(64000/pre['median_ms'],1),
                    'decode_tps':round(8000/dec['median_ms'],1),
                    'state_bytes_after_prefill':state_bytes(st)}
        out['arms'][name]=row
        del m
    return out


if __name__=='__main__':
    result=run()
    dest=Path(__file__).resolve().parents[1]/'research/mathematical-engineering/csl125_r2/NO_FIT_CPU_COMPARISON.json'
    dest.parent.mkdir(parents=True,exist_ok=True)
    dest.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:{'parameters':v['parameters'],'rates':{d:[a['prefill_tps'],a['decode_tps']] for d,a in v['rungs'].items()}}
                      for k,v in result['arms'].items()},indent=2))
