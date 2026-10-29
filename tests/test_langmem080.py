"""LANGMEM080: source-algebra engineered fixture falsifiers (not model skill)."""
from dataclasses import replace
from pathlib import Path
import sys
import pytest
import torch

ROOT=Path(__file__).resolve().parents[1]
for n in ("langmem079","langmem078","langscale075","langscale070",
          "langmemory042","context026","langlayer038","langstream076","langbase024"):
    p=str(ROOT/"research/mathematical-engineering"/n)
    if p not in sys.path:sys.path.insert(0,p)
from dual_native import DualNativeTimescaleTAC078
from dual_streaming import DualNativeStreamingTAC079
from scripts.langmem080_event_trace import (
    source_factors,ANCHORS,DELAYS,PAIRS,SOURCE_ERROR,norm
)


def test_original_source_event_geometry_and_prewrite_native_identity():
    torch.manual_seed(8001)
    model=DualNativeTimescaleTAC078().eval()
    e=DualNativeStreamingTAC079(model,"cap_all")
    with torch.no_grad():
        state=e.start(("fixture-only",))
        for t in range(15):
            token=torch.tensor([t+1],dtype=torch.long)
            factors=source_factors(model,state,token)
            old=state
            y,state=e.step(token,state,owners=("fixture-only",))
            for lane,name in (("fast","native"),("slow","slow_native")):
                p=factors[lane]["p"];q=factors[lane]["q"]
                actual=old.tensors[name]@p+q
                assert (actual-state.tensors[name]).abs().max().item()<=SOURCE_ERROR
            assert state.clock==t+1
            assert y.shape==(1,256)
            assert 0<factors["alpha_fast"]<1 and 0<factors["rho_slow"]<1
            assert factors["query"].shape==(1,8)
            assert factors["write_key"].shape==(1,8)
            assert torch.isfinite(factors["value"]).all()


def test_tagged_historical_write_transport_and_read_counterfactual():
    torch.manual_seed(8002)
    model=DualNativeTimescaleTAC078().eval()
    e=DualNativeStreamingTAC079(model)
    with torch.no_grad():
        state=e.start(("fixture",))
        anchors=(3,5)
        stored={}
        for t in range(14):
            token=torch.tensor([t+2],dtype=torch.long)
            factors=source_factors(model,state,token,mode="native")
            before=state
            logits,next_state=e.step(token,state,owners=("fixture",))
            fresh={}
            for (a,lane),old in stored.items():
                fresh[a,lane]=old@factors[lane]["p"]
            if t in anchors:
                for lane in ("fast","slow"):
                    fresh[t,lane]=factors[lane]["q"].clone()
            if t>8:
                for a in anchors:
                    if a>=t:continue
                    for lane,name in (("fast","native"),("slow","slow_native")):
                        old=stored[a,lane];source=fresh[a,lane]
                        altered=replace(before,tensors={**before.tensors,
                                                 name:before.tensors[name]-old})
                        counter,other=e.step(token,altered,owners=("fixture",))
                        assert counter.shape==logits.shape
                        actual=next_state.tensors[name]-other.tensors[name]
                        assert (actual-source).abs().max().item()<SOURCE_ERROR
                        projected=(source@factors["query"].unsqueeze(-1)).squeeze(-1)
                        if lane=="slow":
                            projected*=factors["read_mix_slow"]
                        assert torch.isfinite(projected).all()
            stored=fresh;state=next_state


def test_source_anchors_and_future_protection():
    assert ANCHORS==(32,64,96,128,192)
    assert DELAYS==(1,8,32,64,128,192)
    assert len(PAIRS)==27
    assert all(a+d<320 and d>0 for a,d in PAIRS)
    torch.manual_seed(8003)
    model=DualNativeTimescaleTAC078().eval()
    e=DualNativeStreamingTAC079(model,"cap_all")
    with torch.no_grad():
        state=e.start(("fixture",))
        token=torch.tensor([55],dtype=torch.long)
        p1=source_factors(model,state,token)
        # Original model inputs never include the future next-byte target.
        irrelevant_future_a=21
        irrelevant_future_b=220
        p2=source_factors(model,state,token)
        assert torch.equal(p1["query"],p2["query"])
        assert irrelevant_future_a!=irrelevant_future_b
