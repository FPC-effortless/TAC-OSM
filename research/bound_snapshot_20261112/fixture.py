"""Frozen untrained source fixture only; no dataset or optimizer."""
from dataclasses import fields,is_dataclass
import torch
from tac_osm.csl125_r3 import CSL125R3Config
from tac_osm.csl125_tool_model import CSL125R4
from tac_osm.csl125_memory_precision import enable_precise_memory_


def model(seed,precise=False):
    torch.set_num_threads(1);torch.manual_seed(seed)
    cfg=CSL125R3Config(vocab_size=256,width=32,ff_width=64,layers=4,
        role_count=4,state_width=8,rungs=(2,4),attention_layers=(2,4),
        query_heads=4,kv_heads=2,attention_window=16,correction_rank=4)
    m=CSL125R4(cfg).eval()
    return enable_precise_memory_(m) if precise else m


def context(depth=4,length=17):
    return dict(depth=depth,stream_ids=('fixture',),observed_tokens=length)


def prefix(m,depth=4,length=17):
    s=m.initial_state(1,torch.device('cpu'),torch.float32,depth=depth,stream_ids=('fixture',))
    if length:
        _,s=m(torch.arange(length).reshape(1,-1).remainder(256),depth=depth,state=s,stream_ids=('fixture',))
    return s


def continuation(m,s,depth=4):
    return m(torch.tensor([[73,74,75]]),depth=depth,state=s,stream_ids=('fixture',),
             all_rungs=True,emit_repair_logits=True,candidate_repair_topk=4)


def values(node,path='root'):
    if isinstance(node,torch.Tensor):
        return {path:node}
    if is_dataclass(node):
        parts=[values(getattr(node,f.name),path+'.'+f.name) for f in fields(node)]
    elif isinstance(node,dict):
        parts=[values(v,path+'.'+str(k)) for k,v in node.items()]
    elif isinstance(node,(tuple,list)):
        parts=[values(v,path+'.'+str(i)) for i,v in enumerate(node)]
    else:return {}
    return {k:v for p in parts for k,v in p.items()}
