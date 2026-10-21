"""ME047 numerical checks only; random tensors DO NOT establish capability."""
import pytest
import torch
from tac_osm.me047_hybrid_capabilities import (ARMS,TemporalTokenizer,MacroTeacher,
       CapabilityForecaster,preflight,parse_intent,safety_clip,GOALS,ROLES)


def test_me047_four_complete_models_real_learner_math_contracts():
    x=preflight()
    assert x["status"]=="ME047_SOFTWARE_ONLY_MATH_PASS"
    assert x["full_native_fast32_slow32_assoc8x8"]
    assert x["pretext_uses_88_true_observed_values"]
    assert x["test_data_used"] is False


def test_me047_validation_of_typed_semantic_schema():
    r=parse_intent({"goal":"peak","role":"power","threshold":45.0})
    assert r.goal=="peak" and r.role=="power" and r.threshold==45.0
    for payload in ({"goal":"natural_language","role":"temperature"},
                    {"goal":"mean","role":"secret"},
                    {"goal":"mean","role":"power","threshold":float("nan")},
                    {"goal":"mean","role":"power","unrecognized":4}):
        with pytest.raises(ValueError):parse_intent(payload)
    assert torch.equal(safety_clip(torch.tensor([-7.,3.,10.]),0.,5.),
                       torch.tensor([0.,3.,5.]))


def test_me047_frozen_pretrained_input_and_macro_override_causality():
    torch.manual_seed(47)
    tokenizer=TemporalTokenizer()
    macro=MacroTeacher()
    model=CapabilityForecaster("hybrid_tac",tokenizer,macro)
    x=torch.randn(2,96);r=torch.tensor([1,2]);goal=torch.tensor([0,2])
    with torch.no_grad():
        expected=model.teacher((x-x.mean(-1,keepdim=True))/
             torch.sqrt((x-x.mean(-1,keepdim=True)).square().mean(-1,keepdim=True)+1e-5))
        a=model(x,r,goal)[0]
        b=model(x,r,goal,macro_anchor=expected)[0]
        assert torch.equal(a,b)
        assert model(x,r,goal)[1].shape==(2,)
    assert all(not p.requires_grad for p in tokenizer.parameters())
    assert all(not p.requires_grad for p in macro.parameters())


def test_me047_native_tac_trains_real_conditional_goal_head():
    torch.manual_seed(47)
    model=CapabilityForecaster("hybrid_tac",TemporalTokenizer(),MacroTeacher())
    x=torch.randn(3,96);r=torch.tensor([1,2,4])
    with torch.no_grad():
        a=model(x,r,torch.zeros(3,dtype=torch.long))[1]
        b=model(x,r,torch.ones(3,dtype=torch.long))[1]
        assert not torch.equal(a,b)
    opt=torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),lr=.001)
    target=torch.randn(3,96)
    from tac_osm.me047_hybrid_capabilities import complete_loss
    opt.zero_grad();loss=complete_loss(model,x,target,r,torch.tensor([0,1,2]))
    loss.backward();opt.step()
    assert model.world.slow_anchor.weight.grad is not None
    assert model.goal_head[0].weight.grad is not None
    assert model.teacher.linear.weight.grad is None
