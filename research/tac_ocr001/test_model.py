from dataclasses import replace

import pytest
import torch
from torch.nn import functional as F

from .model import GroundedOCRDecoder, OCRConfig
from .math_reference import attention_reference
from .tokenizer import encode,decode


torch.set_num_threads(1)


def tiny(family):
    torch.manual_seed(19)
    model=GroundedOCRDecoder(OCRConfig(family=family,d_model=16,n_layers=2,n_heads=2,
        ffn_dim=20,feature_dim=8,reference_budget=4,window=4,chunk_size=3,key_dim=4,value_dim=4)).double()
    features=torch.randn(2,11,8,dtype=torch.float64)
    boxes=torch.rand(2,11,4,dtype=torch.float64)
    refs=model.prepare_visual(features,boxes,owners=("a","b"),page_ids=("p1","p1"))
    return model,features,boxes,refs


def test_lossless_tokenizer():
    text="Total ₦12,345.00 — Ìbàdàn 中文\n\tA\x00"
    assert decode(encode(text))==text
    with pytest.raises(ValueError):
        decode([-1])


@pytest.mark.parametrize("family",["tac","transformer","gru"])
def test_chunk_step_and_state_equivalence(family):
    model,_,_,refs=tiny(family)
    tokens=torch.randint(0,259,(2,13))
    with torch.no_grad():
        full,state=model(tokens,refs,owners=("a","b"))
        previous=model.start(("a","b"))
        pieces=[]
        for i in range(tokens.shape[1]):
            y,previous=model.step(tokens[:,i:i+1],refs,owners=("a","b"),state=previous)
            pieces.append(y)
    torch.testing.assert_close(full,torch.cat(pieces,1),rtol=1e-8,atol=1e-9)
    for name,value in state.tensors().items():
        torch.testing.assert_close(value,previous.tensors()[name],rtol=1e-8,atol=1e-9)


@pytest.mark.parametrize("family",["tac","transformer","gru"])
def test_future_poison_visual_sensitivity_and_gradients(family):
    model,features,boxes,refs=tiny(family)
    tokens=torch.randint(0,259,(2,9))
    y,_=model(tokens,refs,owners=("a","b"))
    poison=tokens.clone()
    poison[:,5:]=(poison[:,5:]+71)%259
    other,_=model(poison,refs,owners=("a","b"))
    torch.testing.assert_close(y[:,:5],other[:,:5],rtol=0,atol=0)
    zero=model.prepare_visual(torch.zeros_like(features),boxes,owners=("a","b"),page_ids=("p1","p1"))
    blank,_=model(tokens,zero,owners=("a","b"))
    assert (y-blank).abs().max()>1e-6  # graph sensitivity, not learned grounding
    loss=F.cross_entropy(y.flatten(0,1),torch.randint(0,259,(18,)))
    loss.backward()
    for name,parameter in model.named_parameters():
        assert parameter.grad is not None,name
        assert torch.isfinite(parameter.grad).all(),name
        assert parameter.grad.abs().sum()>0,name


@pytest.mark.parametrize("family",["tac","transformer","gru"])
def test_bounded_physical_state_and_ownership(family):
    model,features,boxes,refs=tiny(family)
    tokens=torch.randint(0,259,(2,10))
    with torch.inference_mode():
        _,state=model(tokens,refs,owners=("a","b"))
        initial_bytes=state.tensor_bytes()
        before={k:v.clone() for k,v in state.tensors().items()}
        _,final=model(tokens.repeat(1,10),refs,owners=("a","b"),state=state)
    assert final.tensor_bytes()==initial_bytes
    for name,value in final.tensors().items():
        assert value.untyped_storage().nbytes()==value.numel()*value.element_size(),name
        torch.testing.assert_close(before[name],state.tensors()[name],rtol=0,atol=0)
    assert refs.keys.shape[2]==4
    for value in (refs.keys,refs.values):
        assert value.untyped_storage().nbytes()==value.numel()*value.element_size()
    with pytest.raises(ValueError,match="another"):
        model(tokens,refs,owners=("c","d"),state=state)
    with pytest.raises(ValueError,match="another"):
        model(tokens,refs,owners=("a","b"),state=replace(state,signature="other"))
    with pytest.raises(ValueError,match="another"):
        model(tokens,replace(refs,model_tag="other"),owners=("a","b"))


@pytest.mark.parametrize("family",["tac","transformer","gru"])
def test_batch_isolation_and_metadata_blinding(family):
    model,features,boxes,refs=tiny(family)
    tokens=torch.randint(0,259,(2,5))
    with torch.no_grad():
        full,_=model(tokens,refs,owners=("a","b"))
        for i in range(2):
            r=model.prepare_visual(features[i:i+1],boxes[i:i+1],owners=("blinded",),page_ids=("unknown",))
            single,_=model(tokens[i:i+1],r,owners=("blinded",))
            torch.testing.assert_close(full[i:i+1],single,rtol=1e-8,atol=1e-9)


def test_independent_cross_attention_and_derivative():
    torch.manual_seed(2)
    q=torch.randn(3,4,dtype=torch.float64,requires_grad=True)
    k=torch.randn(5,4,dtype=torch.float64)
    v=torch.randn(5,6,dtype=torch.float64)
    output=F.scaled_dot_product_attention(q[None,None],k[None,None],v[None,None])[0,0]
    expected=attention_reference(q.tolist(),k.tolist(),v.tolist())
    torch.testing.assert_close(output,torch.tensor(expected,dtype=torch.float64),rtol=1e-12,atol=1e-12)
    output.sum().backward()
    eps=1e-6
    plus,minus=q.detach().clone(),q.detach().clone()
    plus[1,2]+=eps
    minus[1,2]-=eps
    total=lambda x:sum(map(sum,attention_reference(x.tolist(),k.tolist(),v.tolist())))
    assert float(q.grad[1,2])==pytest.approx((total(plus)-total(minus))/(2*eps),abs=1e-8)


def test_original_active_parameter_match():
    from .presets import matched_presets
    counts=[GroundedOCRDecoder(c).counts()["trainable"] for c in matched_presets().values()]
    assert max(counts)/min(counts)-1<.01
    assert counts==[1309376,1309312,1309824]


@pytest.mark.parametrize("family",["tac","transformer","gru"])
def test_source_probe_denies_label_path(family):
    from .probe_adapter import adapter_for
    model,features,boxes,_=tiny(family)
    adapter=adapter_for(model)
    sample={"history":torch.tensor([[256,5],[256,8]]),"query":torch.tensor([[6],[7]]),
            "owners":("a","b"),"page_ids":("p1","p1"),"features":features,"boxes":boxes}
    adapter.validate(model,sample)
    state=adapter.advance(model,adapter.start(model,sample),sample,0)
    assert adapter.forecast(model,state,sample)["next_byte_logits"].shape==(2,1,259)
    with pytest.raises(ValueError,match="denies"):
        adapter.validate(model,{**sample,"gold_text":"not an inference input"})
