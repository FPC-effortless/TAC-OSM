"""Numerical unit checks, not generated-data capability experiments."""
import pytest
torch=pytest.importorskip('torch')
from tac_osm.me035_residual_forecast import ARMS,build_model


@pytest.mark.parametrize('arm',ARMS)
def test_equal_active_capacity_and_script(arm):
    m=build_model(arm,35001);x=torch.randn(2,96,7)
    assert sum(p.numel() for p in m.parameters())==18924
    s=torch.jit.script(m)
    torch.testing.assert_close(s(x),m(x),rtol=0,atol=1e-7)


def test_normalization_stage_starts_with_identical_learned_weights():
    a,b=[build_model(arm,35001) for arm in ARMS[:2]]
    assert all(torch.equal(v,b.state_dict()[k]) for k,v in a.state_dict().items())


def test_residual_slow_has_exact_identity_jacobian_at_zero_innovation():
    m=build_model('native_residual',35001)
    with torch.no_grad():m.slow.weight.zero_();m.slow.bias.zero_()
    u=torch.zeros(1,7);state=torch.zeros(1,64,requires_grad=True)
    j=torch.autograd.functional.jacobian(lambda s:m.advance(u,s)[:,32:],state).reshape(32,64)
    torch.testing.assert_close(j[:,32:],torch.eye(32),rtol=0,atol=1e-7)


@pytest.mark.parametrize('arm',['native_joint','native_multiscale'])
def test_zero_forecast_residual_exactly_preserves_last_observation(arm):
    m=build_model(arm,35001)
    with torch.no_grad():
        for layer in (m.direct,m.output):layer.weight.zero_();layer.bias.zero_()
    x=torch.randn(3,96,7)
    torch.testing.assert_close(m(x),x[:,-1:].expand(-1,3,-1),rtol=0,atol=0)


def test_constant_history_is_finite_and_all_body_paths_receive_gradients():
    m=build_model('native_multiscale',35001)
    assert torch.isfinite(m(torch.ones(2,96,7))).all()
    loss=m(torch.randn(3,96,7)).square().mean();loss.backward()
    for name,p in m.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum()>0,name


def test_native_state_remains_bounded_for_residual_slow_updates():
    m=build_model('native_multiscale',35001);s=torch.zeros(2,64)
    with torch.no_grad():
        for _ in range(150):s=m.advance(torch.randn(2,7)*5,s)
    assert torch.isfinite(s).all() and s.abs().max()<=1.


def test_multiscale_initial_rates_and_transition_gain():
    m=build_model('native_multiscale',35001)
    torch.testing.assert_close(m.gate.bias.sigmoid()[[0,-1]],torch.tensor([.5,1/32]),rtol=1e-6,atol=1e-7)
    torch.testing.assert_close(m.slow_gate.bias.sigmoid()[[0,-1]],torch.tensor([1/32,1/256]),rtol=1e-6,atol=1e-7)
    torch.testing.assert_close(m.transition.weight.T@m.transition.weight,torch.eye(32)*.81,rtol=0,atol=5e-7)


def test_invalid_shape_rejected():
    with pytest.raises(ValueError,match='96x7'):build_model('native_joint',35001)(torch.zeros(2,97,7))
