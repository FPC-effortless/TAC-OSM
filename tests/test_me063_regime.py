"""ME063 source-contract tests: synthetic software diagnostics ONLY."""
import torch
from tests.test_me057_training_contract import identified
from tac_osm.real_robot_me063_regime import RegimeMixtureModel, gaussian_physical_loss

def make(name):
    width, decoder = (32, 64) if name=="native_tac" else (44,82)
    return RegimeMixtureModel(identified(),name,width,decoder)

def test_causal_mixture_validity_and_finite_uncertainty():
    torch.manual_seed(63001)
    current=torch.randn(2,14)
    actions=torch.randn(2,16,14)
    for arm in ("native_tac","gru_world"):
        model=make(arm)
        belief=model.initial(2)
        old=model.forecast(belief,current,actions)
        assert old.mean.shape==(2,16,14)
        assert old.scale.shape==(2,16,14)
        assert old.weights.shape==(2,16,3)
        assert torch.isfinite(old.mean).all()
        assert torch.isfinite(old.scale).all() and (old.scale>0).all()
        torch.testing.assert_close(old.weights.sum(-1),torch.ones(2,16),rtol=0,atol=1e-6)
        new_actions=actions.clone()
        new_actions[:,5:]+=100
        new=model.forecast(belief,current,new_actions)
        torch.testing.assert_close(old.mean[:,:5],new.mean[:,:5],rtol=0,atol=0)
        torch.testing.assert_close(old.scale[:,:5],new.scale[:,:5],rtol=0,atol=0)

def test_factual_gradient_paths_under_real_shape_contract():
    torch.manual_seed(63002)
    states=torch.randn(2,5,14)
    executed=torch.randn(2,4,14)
    future_actions=torch.randn(2,16,14)
    actual=torch.randn(2,16,14)
    for arm in ("native_tac","gru_world"):
        model=make(arm)
        belief=model.initial(2)
        for t in range(4):
            belief=model.observe(belief,states[:,t],executed[:,t],states[:,t+1],torch.zeros(2,1,dtype=torch.bool))
        forecast=model.forecast(belief,states[:,-1],future_actions)
        loss=gaussian_physical_loss(model,forecast,actual,torch.ones(2,16,1,dtype=torch.bool))
        assert torch.isfinite(loss)
        loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
