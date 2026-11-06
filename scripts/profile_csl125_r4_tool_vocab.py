"""No-fit vocab/weight-packing capacity screen for R4; no tokenizer trained.

Counts the full 20-layer model including all heads. This is NOT a CQ2 export,
runtime memory, model intelligence or token-efficiency benchmark.
"""
from dataclasses import replace
from math import ceil
import json

import torch

from tac_osm.csl125_r3 import CSL125R3Config
from tac_osm.csl125_tool_model import CSL125R4


def capacity_screen():
    result={"stage":"NO_FIT_META_PARAMETERS_ONLY","optimizer_steps":0,
            "bpe_fitted":False,"model_quality":"NOT_MEASURED",
            "token_lengths":"NOT_MEASURED","quantized_runtime":"NOT_BUILT",
            "vocabularies":{}}
    for v in (8192,16384,32768):
        cfg=replace(CSL125R3Config(),vocab_size=v)
        with torch.device('meta'):
            model=CSL125R4(cfg,device='meta')
        p=model.parameter_count()
        result["vocabularies"][str(v)]={
            "params":p,"embedding_params":v*cfg.width,
            "ideal_2bit_weight_bytes":ceil(p*2/8),
            "full_vocab_projection_Mac_per_token":v*cfg.width,
            "headroom_to_125m":125_000_000-p,
            "hidden_width":cfg.width,"layers":cfg.layers,
        }
    print(json.dumps(result,indent=2,sort_keys=True))
    return result


if __name__=="__main__":
    capacity_screen()
