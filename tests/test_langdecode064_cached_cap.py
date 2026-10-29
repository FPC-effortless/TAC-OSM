"""LANGDECODE064 constructed parity and cap boundary, not language evidence."""
import inspect
from dataclasses import replace

import pytest
import torch
from scripts.langdecode064_cached_cap import CachedCapPositionTAC064
from scripts.langdecode057_validation import BatchedValidationTAC057
from incremental import IncrementalTAC043, SHAPES, phi_position, transform
from factorial_depth import FactorialDepthControl040


@pytest.mark.parametrize("dtype",[torch.float32,torch.float64])
def test_cached_position_is_original_transform_at_all_clock_boundaries(dtype):
    model=FactorialDepthControl040(state_width=48,upper_graph=False).to(dtype).eval()
    candidate=CachedCapPositionTAC064(model,"cap_all")
    with torch.inference_mode():
        for t in (128,129,512,8192,65536):
            pos=transform(phi_position(torch.tensor([t],dtype=torch.long),dtype),"cap_all")
            assert torch.equal(pos,candidate._cap_all_ref), t
        assert candidate._cap_all_ref.shape==(1,7)
        assert candidate._cap_all_ref.dtype==dtype
        # t=127 is uncapped according to the original source (strict >).
        at127=transform(phi_position(torch.tensor([127],dtype=torch.long),dtype),"cap_all")
        assert torch.equal(at127,candidate._cap_all_ref)
        assert candidate._cap_all_ref.numel()==7


@pytest.mark.parametrize("dtype",[torch.float32,torch.float64])
@pytest.mark.parametrize("mode",["native","cap_all"])
def test_two_owner_masked_bitwise_original_logits_and_state(dtype,mode):
    torch.manual_seed(64064)
    model=FactorialDepthControl040(state_width=48,upper_graph=False).to(dtype).eval()
    original=BatchedValidationTAC057(model,mode)
    candidate=CachedCapPositionTAC064(model,mode)
    owners=("a","b")
    source=torch.randint(0,256,(2,132),dtype=torch.long)
    mask=(torch.arange(132)[None,:]+torch.arange(2)[:,None])%3 != 1
    with torch.inference_mode():
        y,so=original.prefill(source[:,:127],owners=owners,write_mask=mask[:,:127])
        z,sc=candidate.prefill(source[:,:127],owners=owners,write_mask=mask[:,:127])
        assert torch.equal(y,z)
        assert all(torch.equal(so.tensors[k],sc.tensors[k]) for k in SHAPES)
        for t in range(127,132):
            y,so=original.step(source[:,t],so,owners=owners,write_mask=mask[:,t])
            z,sc=candidate.step(source[:,t],sc,owners=owners,write_mask=mask[:,t])
            assert torch.equal(y,z),t
            assert so.clock==sc.clock==t+1
            assert all(torch.equal(so.tensors[k],sc.tensors[k]) for k in SHAPES),t
        expected=2*976*torch.tensor([],dtype=dtype).element_size()
        assert so.tensor_bytes()==sc.tensor_bytes()==expected


def test_snapshot_interoperability_and_same_public_decoder_api(tmp_path):
    torch.manual_seed(64065)
    model=FactorialDepthControl040(state_width=48,upper_graph=False).eval()
    original=BatchedValidationTAC057(model,"cap_all")
    candidate=CachedCapPositionTAC064(model,"cap_all")
    owners=("source",)
    with torch.inference_mode():
        source=torch.randint(0,256,(1,128),dtype=torch.long)
        yo,so=original.prefill(source,owners=owners)
        yc,sc=candidate.prefill(source,owners=owners)
        assert torch.equal(yo,yc)
        for i,(writer,reader,s) in enumerate(((original,candidate,so),(candidate,original,sc))):
            path=tmp_path/f"state-{i}.pt"
            writer.save_state(s,path)
            recovered=reader.load_state(path,owners=owners)
            y,sy=reader.step(torch.tensor([123]),recovered,owners=owners)
            z,sz=writer.step(torch.tensor([123]),s,owners=owners)
            assert torch.equal(y,z)
            assert all(torch.equal(sy.tensors[k],sz.tensors[k]) for k in SHAPES)
        with pytest.raises(ValueError):
            candidate.step(torch.tensor([65]),replace(sc,clock=-1),owners=owners)
    assert CachedCapPositionTAC064.step is IncrementalTAC043.step
    assert CachedCapPositionTAC064.validate is BatchedValidationTAC057.validate
    assert CachedCapPositionTAC064.prefill is IncrementalTAC043.prefill


def test_candidate_is_source_copy_with_one_position_substitution():
    ref=inspect.getsource(IncrementalTAC043._advance)
    cand=inspect.getsource(CachedCapPositionTAC064._advance)
    assert "m.pos(transform(phi_position(tt,embed.dtype),self.position_mode))" in ref
    assert "m.pos(transform(phi_position(tt,embed.dtype),self.position_mode))" not in cand
    assert 'self.position_mode=="cap_all" and t>=128' in cand
    for unchanged in ("ordinal_code(tt,e.dtype)","torch.linalg.solve",
                      "upper.read_from_prior","upper.observe","m.ffn_out",
                      "s['native']@native_p+native_q"):
        assert unchanged in cand and unchanged in ref
