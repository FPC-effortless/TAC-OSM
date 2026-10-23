"""Numerical-only tests. These do not establish real-world model capability."""
import torch
import pytest

from tac_osm.me065_micro_world_controls import (
    GaussianRSSMCell, SelectiveDiagonalCell, MicroRegimeComparator)
from tac_osm.real_robot_me063_regime import gaussian_physical_loss


def source_identification():
    one = torch.ones(14)
    zero = torch.zeros(14)
    return dict(state_mean=zero, state_scale=one,
                action_mean=zero, action_scale=one,
                delta_scale=one * .1,
                servo_gain=one * .1, servo_step_limit=one * .1)


@pytest.mark.parametrize("variant,width,decoder", [
    ("compact_rssm", 44, 82), ("selective_diagonal_ssm", 44, 103)])
def test_observation_and_action_forecast_are_finite(variant,width,decoder):
    torch.manual_seed(65)
    model=MicroRegimeComparator(source_identification(),variant,width,decoder)
    observation=model.initial(2)
    assert observation.fast.shape == (2,width)
    model.train()
    model.reset_auxiliary()
    previous=torch.zeros(2,14)
    for i in range(3):
        current=previous+.002
        observation=model.observe(observation,previous,
                                  torch.ones(2,14)*.03,current,
                                  torch.zeros(2,1,dtype=torch.bool))
        previous=current
    prediction=model.forecast(observation,previous,torch.full((2,4,14),.02))
    assert prediction.mean.shape==(2,4,14)
    assert prediction.scale.shape==(2,4,14)
    assert torch.isfinite(prediction.mean).all()
    assert torch.isfinite(prediction.scale).all()
    assert torch.all(prediction.scale>0)
    assert torch.allclose(prediction.weights.sum(-1),torch.ones(2,4),atol=1e-6)
    loss=gaussian_physical_loss(model,prediction,previous[:,None].expand(-1,4,-1),
                                torch.ones(2,4,1,dtype=torch.bool))
    loss=loss+model.auxiliary_objective()
    loss.backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    assert any(p.grad is not None for p in model.parameters())


def test_rssm_kl_is_trained_and_eval_is_deterministic():
    torch.manual_seed(65)
    cell=GaussianRSSMCell(24,44,8)
    cell.train()
    x=torch.randn(2,24)
    h=torch.zeros(2,44)
    cell.clear_kl()
    next_state=cell(x,h)
    assert next_state.shape==(2,44)
    assert torch.isfinite(cell.regularizer())
    assert cell.regularizer().item()>=0.
    cell.eval()
    with torch.no_grad():
        a=cell(x,h)
        b=cell(x,h)
    assert torch.equal(a,b)


def test_selective_state_is_finite_over_long_numerical_stream():
    torch.manual_seed(65)
    cell=SelectiveDiagonalCell(24,44)
    state=torch.zeros(3,44)
    for _ in range(128):
        state=cell(torch.randn(3,24),state)
    assert torch.isfinite(state).all()


def test_unknown_control_rejected():
    with pytest.raises(ValueError):
        MicroRegimeComparator(source_identification(),"not_a_comparator")
