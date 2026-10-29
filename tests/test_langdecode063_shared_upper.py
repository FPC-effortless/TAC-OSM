"""LANGDECODE063 constructed exactness checks, not original-model evidence."""
import inspect
from dataclasses import replace

import pytest
import torch
from scripts.langdecode057_validation import BatchedValidationTAC057
from scripts.langdecode063_shared_upper import SharedUpperTAC063
from incremental import IncrementalTAC043, SHAPES
from factorial_depth import FactorialDepthControl040


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("mode", ["native", "cap_all"])
def test_two_owner_masked_equivalence(dtype, mode):
    torch.manual_seed(63063)
    model=FactorialDepthControl040(state_width=48,upper_graph=False).to(dtype).eval()
    original=BatchedValidationTAC057(model,mode)
    candidate=SharedUpperTAC063(model,mode)
    owners=("alpha","beta")
    data=torch.randint(0,256,(2,16),dtype=torch.long)
    mask=(torch.arange(16)[None,:]+torch.arange(2)[:,None])%3!=0
    with torch.inference_mode():
        yo,so=original.prefill(data[:,:8],owners=owners,write_mask=mask[:,:8])
        yc,sc=candidate.prefill(data[:,:8],owners=owners,write_mask=mask[:,:8])
        assert torch.equal(yo,yc)
        assert so.clock==sc.clock==8
        assert all(torch.equal(so.tensors[k],sc.tensors[k]) for k in SHAPES)
        for t in range(8,16):
            yo,so=original.step(data[:,t],so,owners=owners,write_mask=mask[:,t])
            yc,sc=candidate.step(data[:,t],sc,owners=owners,write_mask=mask[:,t])
            assert torch.equal(yo,yc)
            assert so.clock==sc.clock==t+1
            assert all(torch.equal(so.tensors[k],sc.tensors[k]) for k in SHAPES)
        assert so.tensor_bytes()==sc.tensor_bytes()==2*976*torch.tensor([],dtype=dtype).element_size()


def test_upper_read_observe_reuse_in_isolation_exact():
    torch.manual_seed(63064)
    for dtype in (torch.float32,torch.float64):
        model=FactorialDepthControl040(state_width=48,upper_graph=False).to(dtype).eval()
        ublock=model.layernorm[1]
        z=torch.randn(2,280,dtype=dtype)
        prior=torch.randn(2,48,dtype=dtype)
        with torch.inference_mode():
            original_read=ublock.read_from_prior(z,prior)
            original_write=ublock.observe(z,prior)
            u=ublock.input_proj(ublock.input_norm(z)).tanh()
            gate=ublock.read_gate(torch.cat((u,prior),-1)).sigmoid()
            fused_read=z+ublock.residual_gain.sigmoid()*ublock.output((u+gate*prior).tanh())
            alpha=.995*ublock.forget(u).sigmoid()
            fused_write=alpha*prior+(1-alpha)*ublock.offer(u).tanh()
        assert torch.equal(original_read,fused_read)
        assert torch.equal(original_write,fused_write)


def test_snapshots_bidirectional_and_guards(tmp_path):
    model=FactorialDepthControl040(state_width=48,upper_graph=False).eval()
    original=BatchedValidationTAC057(model,"cap_all")
    candidate=SharedUpperTAC063(model,"cap_all")
    owners=("owner",)
    with torch.inference_mode():
        tokens=torch.randint(0,256,(1,16))
        _,so=original.prefill(tokens,owners=owners)
        _,sc=candidate.prefill(tokens,owners=owners)
        for i,(writer,reader,state) in enumerate(((original,candidate,so),(candidate,original,sc))):
            path=tmp_path/f"state-{i}.pt"
            writer.save_state(state,path)
            restored=reader.load_state(path,owners=owners)
            assert all(torch.equal(state.tensors[k],restored.tensors[k]) for k in SHAPES)
            y,next_a=reader.step(torch.tensor([65]),restored,owners=owners)
            z,next_b=writer.step(torch.tensor([65]),state,owners=owners)
            assert torch.equal(y,z)
            assert all(torch.equal(next_a.tensors[k],next_b.tensors[k]) for k in SHAPES)
        with pytest.raises(ValueError):
            candidate.step(torch.tensor([65]),replace(sc,clock=-1),owners=owners)


def test_original_transition_and_public_methods_unchanged():
    original=inspect.getsource(IncrementalTAC043._advance)
    candidate=inspect.getsource(SharedUpperTAC063._advance)
    assert "upper.read_from_prior(z,s['upper']);next_upper=upper.observe(z,s['upper'])" in original
    assert "upper.input_proj(upper.input_norm(z)).tanh()" in candidate
    assert "upper.read_from_prior" not in candidate and "upper.observe(" not in candidate
    assert candidate.count("upper.input_proj(")==1
    assert "ordinal_code(tt,e.dtype)" in candidate
    assert "torch.linalg.solve" in candidate
    assert "m.ffn_out" in candidate
    assert SharedUpperTAC063.step is IncrementalTAC043.step
    assert SharedUpperTAC063.prefill is IncrementalTAC043.prefill
    assert SharedUpperTAC063.validate is BatchedValidationTAC057.validate
