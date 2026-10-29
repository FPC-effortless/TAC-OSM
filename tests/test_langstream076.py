"""LANGSTREAM076: constructed real carried-state schema, original source parity.

These are *engineering tests*, not evidence of genuine language-model skill.
"""
from dataclasses import replace
from pathlib import Path
import sys
import pytest
import torch

ROOT=Path(__file__).resolve().parents[1]
for n in ("langstream076","langscale075","langscale070","langlayer040"):
    p=str(ROOT/"research/mathematical-engineering"/n)
    if p not in sys.path:sys.path.insert(0,p)
from native_streaming import NativeStreamingTAC076,state_shapes
from native_matrix import NativeMatrixTAC075
from factorial_depth import FactorialDepthControl040
from incremental import IncrementalTAC043


@pytest.mark.parametrize("tier,k,state_bytes",[("m8",8,3904),("m12",12,4752),("m16",16,5728)])
def test_true_carried_prefill_and_next_step_match_original_growing_source(tier,k,state_bytes,tmp_path):
    torch.manual_seed(76001)
    model=NativeMatrixTAC075(tier).eval()
    engine=NativeStreamingTAC076(model,"cap_all")
    tokens=torch.randint(0,256,(2,19),dtype=torch.long)
    owners=("original-publisher-a","original-publisher-b")
    with torch.inference_mode():
        prefix=engine.prefill(tokens[:,:16],owners=owners)
        y,state=prefix
        assert state.clock==16 and state.tensor_bytes()==2*state_bytes
        assert state.tensors["native"].shape==(2,k,k)
        assert state.tensors["aligned"].shape==(2,2,16,k)
        assert state.tensors["previous_key"].shape==(2,k)
        original=model(tokens[:,:16])[:,-1]
        assert (y-original).abs().max().item()<=3e-5
        for t in range(16,19):
            y,state=engine.step(tokens[:,t],state,owners=owners)
            assert state.clock==t+1
            correct=model(tokens[:,:t+1])[:,-1]
            assert (correct-y).abs().max().item()<=3e-5
        p=tmp_path/f"state_{tier}.pt"
        engine.save_state(state,p)
        reread=engine.load_state(p,owners=owners)
        assert reread.clock==state.clock
        assert reread.tensor_bytes()==2*state_bytes
        assert all(torch.equal(reread.tensors[key],state.tensors[key]) for key in state.tensors)
    with pytest.raises(ValueError):
        engine.load_state(p,owners=tuple(reversed(owners)))


def test_original_m8_source_matches_original_043_incremental_step_bitwise():
    torch.manual_seed(76002)
    original=FactorialDepthControl040(state_width=48,upper_graph=False).eval()
    torch.manual_seed(76002)
    same=NativeMatrixTAC075("m8").eval()
    assert tuple(original.state_dict())==tuple(same.state_dict())
    assert all(torch.equal(original.state_dict()[n],same.state_dict()[n])
               for n in original.state_dict())
    old=IncrementalTAC043(original,"native")
    new=NativeStreamingTAC076(same,"native")
    tokens=torch.randint(0,256,(2,13),dtype=torch.long)
    mask=torch.ones(2,13,dtype=torch.bool);mask[1,4]=False;mask[0,7]=False
    oo=("a","b")
    with torch.inference_mode():
        os=old.start(oo);ns=new.start(oo)
        for i in range(13):
            loy,os=old.step(tokens[:,i],os,owners=oo,write_mask=mask[:,i])
            lny,ns=new.step(tokens[:,i],ns,owners=oo,write_mask=mask[:,i])
            assert torch.equal(loy,lny)
            assert os.clock==ns.clock
            assert all(torch.equal(os.tensors[k],ns.tensors[k]) for k in os.tensors)


def test_same_clock_native_memory_lesion_does_not_mutate_original_state():
    torch.manual_seed(76003)
    model=NativeMatrixTAC075("m16").eval()
    e=NativeStreamingTAC076(model,"cap_all")
    owners=("owner",)
    with torch.inference_mode():
        state=e.prefill(torch.arange(32,dtype=torch.long).view(1,32),owners=owners)[1]
        before=state.tensors["native"].clone()
        lesion=replace(state,tensors={**state.tensors,
                                       "native":torch.zeros_like(state.tensors["native"])})
        y,correct=e.step(torch.tensor([32]),state,owners=owners)
        q,ablated=e.step(torch.tensor([32]),lesion,owners=owners)
        assert y.shape==q.shape==(1,256)
        assert correct.clock==ablated.clock==33
        assert torch.isfinite(q).all()
        assert torch.equal(state.tensors["native"],before)
        assert not torch.equal(lesion.tensors["native"],before)


def test_unregistered_native_state_shape_fails_closed():
    for k in (0,4,10,24,True,16.0):
        with pytest.raises(ValueError):state_shapes(k)
