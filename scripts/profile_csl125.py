"""CSL-125 meta-device parameter plan, no training/model weights allocated."""
from __future__ import annotations
import json
from math import ceil
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import torch
from tac_osm.csl125 import CAP,CSL125,CSL125Config


def main():
    cfg=CSL125Config()
    with torch.device('meta'):
        model=CSL125(cfg,device='meta')
    total=model.parameter_count()
    groups=model.parameter_breakdown()
    assert total==123699534 and total<=CAP
    result={
      'variant':'CSL-125',
      'status':'NO_FIT_NO_WEIGHTS_UTILITY_ONLY',
      'vocab_size':cfg.vocab_size,
      'tokenizer_status':'PROPOSED_BYTE_FALLBACK_BPE_NOT_FITTED',
      'width':cfg.width,'layers':cfg.layers,
      'rungs':cfg.rungs,'role_slots':cfg.role_count,
      'state_width':cfg.state_width,
      'local_attention_layers':cfg.attention_layers,
      'attention_window':cfg.attention_window,
      'query_heads':cfg.query_heads,
      'kv_heads':cfg.kv_heads,
      'parameter_groups':groups,
      'instantiated_parameters':total,
      'parameter_cap':CAP,
      'headroom':CAP-total,
      'weights_mib_ideal_lower_bound':{
        str(b):round(ceil(total*b/8)/(1024*1024),3)
        for b in (32,16,8,4,2.125)},
      'optimizer_steps':0,
      'warnings':['2.125-bit entry is an unachieved packing lower bound, not exported CQ2',
                  'No tokenizer model, training, quantized inference, memory RSS, FLOPs, latency, quality or MCU results',
                  'TAC-OSM future training policy currently requires 100k+/-5%; no exception or compute approval'],
    }
    print(json.dumps(result,indent=2))
    return result

if __name__=='__main__':
    main()
