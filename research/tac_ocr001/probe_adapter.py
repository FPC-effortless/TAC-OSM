"""Actual OCR graph adapters; no borrowed robot/native state assumptions."""
from __future__ import annotations

from dataclasses import replace

import torch

from tac_osm.research_probe import ResearchProbeAdapter
from research.langcache104.model import LayerState104
from .model import GroundedOCRDecoder


def patch_state(state,name,value):
    if name not in state.tensors():
        raise ValueError("unknown OCR state field")
    _,idx,field=name.split(".")
    layers=list(state.layers)
    idx=int(idx)
    old=layers[idx]
    if isinstance(old,LayerState104):
        layers[idx]=replace(old,**{field:value})
    elif isinstance(old,torch.Tensor):
        layers[idx]=value
    else:
        layers[idx]=tuple(value if key==field else old[i] for i,key in enumerate(("key","value")))
    return replace(state,layers=tuple(layers))


def adapter_for(model):
    if not isinstance(model,GroundedOCRDecoder):
        raise ValueError("actual grounded OCR source model required")

    def validate(model,sample):
        if set(sample)!={"history","query","owners","page_ids","features","boxes"}:
            raise ValueError("source input boundary denies labels/metadata/oracle fields")
        if sample["history"].ndim!=2 or sample["query"].shape!=(sample["history"].shape[0],1):
            raise ValueError("legal observed prefix and single-byte query required")
        references(model,sample)

    def references(model,sample):
        return model.prepare_visual(sample["features"],sample["boxes"],owners=sample["owners"],page_ids=sample["page_ids"])

    def forecast(model,state,sample):
        return {"next_byte_logits":model(sample["query"],references(model,sample),owners=sample["owners"],state=state,mode="recurrent")[0]}

    def gradient_graph(model,state,sample):
        leaves={name:value.detach().clone().requires_grad_() for name,value in state.tensors().items() if value.numel()}
        altered=state
        for name,value in leaves.items():
            altered=patch_state(altered,name,value)
        return forecast(model,altered,sample),leaves

    return ResearchProbeAdapter(
        name="tac-ocr001-"+model.config.family,
        validate=validate,
        steps=lambda sample:sample["history"].shape[1],
        start=lambda model,sample:model.start(sample["owners"]),
        advance=lambda model,state,sample,t:model(sample["history"][:,t:t+1],references(model,sample),
            owners=sample["owners"],state=state,mode="recurrent")[1],
        forecast=forecast,state_views=lambda state:state.tensors(),patch_state=patch_state,
        batch_size=lambda sample:sample["history"].shape[0],
        batch_item=lambda sample,i:{"history":sample["history"][i:i+1],"query":sample["query"][i:i+1],
            "owners":(sample["owners"][i],),"page_ids":(sample["page_ids"][i],),
            "features":sample["features"][i:i+1],"boxes":sample["boxes"][i:i+1]},
        gradient_graph=gradient_graph,permute_candidates=None)
