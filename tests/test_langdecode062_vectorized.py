"""Constructed regression, not original-language performance evidence."""
import inspect
import pytest
import torch
from dataclasses import replace

from scripts.langdecode062_vectorized import VectorOrdinalTAC062
from scripts.langdecode057_validation import BatchedValidationTAC057
from incremental import IncrementalTAC043,SHAPES
from factorial_depth import FactorialDepthControl040


@pytest.mark.parametrize("dtype",[torch.float32,torch.float64])
@pytest.mark.parametrize("mode",["native","cap_all"])
def test_exact_two_owner_causal_masked_state_and_logits(dtype,mode):
    torch.manual_seed(62062)
    model=FactorialDepthControl040(state_width=48,upper_graph=False).to(dtype).eval()
    reference=BatchedValidationTAC057(model,mode)
    candidate=VectorOrdinalTAC062(model,mode)
    owners=("owner-a","owner-b")
    tokens=torch.randint(0,256,(2,132))
    times=torch.arange(132)
    mask=(times[None,:]+torch.arange(2)[:,None])%4!=0
    with torch.inference_mode():
        y,st=reference.prefill(tokens[:,:128],owners=owners,write_mask=mask[:,:128])
        z,other=candidate.prefill(tokens[:,:128],owners=owners,write_mask=mask[:,:128])
        assert torch.equal(y,z)
        assert st.clock==other.clock==128
        assert all(torch.equal(st.tensors[k],other.tensors[k]) for k in SHAPES)
        for t in range(128,132):
            y,st=reference.step(tokens[:,t],st,owners=owners,write_mask=mask[:,t])
            z,other=candidate.step(tokens[:,t],other,owners=owners,write_mask=mask[:,t])
            assert torch.equal(y,z)
            assert all(torch.equal(st.tensors[k],other.tensors[k]) for k in SHAPES)
        assert st.tensor_bytes()==other.tensor_bytes()==7808


def test_original_snapshot_roundtrip_both_directions(tmp_path):
    torch.manual_seed(62063)
    model=FactorialDepthControl040(state_width=48,upper_graph=False).eval()
    a=BatchedValidationTAC057(model,"cap_all")
    b=VectorOrdinalTAC062(model,"cap_all")
    tokens=torch.randint(0,256,(1,128))
    with torch.inference_mode():
        _,s=a.prefill(tokens,owners=("a",))
        _,q=b.prefill(tokens,owners=("a",))
        assert all(torch.equal(s.tensors[k],q.tensors[k]) for k in SHAPES)
        p=tmp_path/"state-original.pt"
        a.save_state(s,p)
        re=b.load_state(p,owners=("a",))
        assert all(torch.equal(s.tensors[k],re.tensors[k]) for k in SHAPES)
        path2=tmp_path/"state-candidate.pt"
        b.save_state(q,path2)
        restore=a.load_state(path2,owners=("a",))
        for state in (re,restore):
            x,st=b.step(torch.tensor([121]),state,owners=("a",))
            y,sr=a.step(torch.tensor([121]),s,owners=("a",))
            assert torch.equal(x,y)
            assert all(torch.equal(st.tensors[k],sr.tensors[k]) for k in SHAPES)


def test_full_step_guards_and_no_protected_state_changes():
    model=FactorialDepthControl040(state_width=48,upper_graph=False).eval()
    candidate=VectorOrdinalTAC062(model,"cap_all")
    state=candidate.start(("a",))
    for bad in (replace(state,clock=-1),replace(state,tensors={})):
        with pytest.raises(ValueError):
            candidate.step(torch.tensor([65]),bad,owners=("a",))
    assert VectorOrdinalTAC062.step is IncrementalTAC043.step
    assert VectorOrdinalTAC062.validate is BatchedValidationTAC057.validate
    assert VectorOrdinalTAC062.prefill is IncrementalTAC043.prefill


def test_candidate_carries_full_original_source_except_ordinal_kernel():
    a=inspect.getsource(IncrementalTAC043._advance)
    b=inspect.getsource(VectorOrdinalTAC062._advance)
    assert "ordinal_code(tt,e.dtype)" in a and "ordinal_code(tt,e.dtype)" not in b
    assert "vectorized_ordinal" in b
    assert "torch.linalg.solve" in a and "torch.linalg.solve" in b
    assert "m.ffn_out" in a and "m.ffn_out" in b
    assert "s['native']@native_p+native_q" in a and "s['native']@native_p+native_q" in b
