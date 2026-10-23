"""Fabricated numerical/causality cases only; real model evidence is separate."""
import torch
from scripts.me061_train_reliable import build,objective
from tests.test_me057_training_contract import identified


def test_initial_forecast_is_exactly_persistence_for_both_arms():
    current=torch.randn(3,14);actions=torch.randn(3,16,14)*100
    for name in ("native_tac","gru_world"):
        model=build(name,identified())
        p=model.forecast(model.initial(3),current,actions)
        torch.testing.assert_close(p.mean,current[:,None].expand(-1,16,-1),rtol=0,atol=0)
        assert (p.scale>0).all() and torch.isfinite(p.scale).all()


def test_reliability_observer_is_bounded_and_zero_when_evidence_agrees():
    model=build("native_tac",identified());b=model.initial(3)
    current=torch.randn(3,14);a=torch.randn(3,14)
    expected=current+model.servo_step(current,a)
    b=model.observe(b,current,a,expected,torch.zeros(3,1,dtype=torch.bool))
    assert b.innovation.max()<1e-5
    for _ in range(100):
        b=model.observe(b,current,a,current+100,torch.zeros(3,1,dtype=torch.bool))
    assert b.velocity.abs().max()<=1 and b.innovation.max()<=20
    assert b.fast.abs().max()<=1 and b.slow.abs().max()<=1
    assert b.memory.square().sum((-2,-1)).sqrt().max()<1.00001


def test_all_parameters_can_train_and_remain_finite_after_update():
    torch.manual_seed(61001)
    states=torch.randn(8,81,14);actions=torch.randn_like(states)
    flags=torch.zeros(8,81,dtype=torch.bool)
    for name in ("native_tac","gru_world"):
        model=build(name,identified());optimizer=torch.optim.AdamW(model.parameters(),lr=.001)
        for _ in range(2):
            optimizer.zero_grad(set_to_none=True);loss=objective(model,(states,actions,flags))
            loss.backward()
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
            optimizer.step()
        assert all(torch.isfinite(p).all() for p in model.parameters())


def test_later_actions_cannot_affect_earlier_horizons_after_learning():
    model=build("native_tac",identified())
    with torch.no_grad():model.output.weight.normal_(std=.01)
    state=model.initial(2);current=torch.randn(2,14);actions=torch.randn(2,16,14)
    before=model.forecast(state,current,actions)
    actions[:,5:]+=100
    after=model.forecast(state,current,actions)
    torch.testing.assert_close(before.mean[:,:5],after.mean[:,:5],rtol=0,atol=0)
    torch.testing.assert_close(before.scale[:,:5],after.scale[:,:5],rtol=0,atol=0)
