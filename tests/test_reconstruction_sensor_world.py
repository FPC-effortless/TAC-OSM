import math

import pytest
import torch
import torch.nn.functional as F

from tac_osm.variants.reconstruction_sensor_world import ReconstructionSensorWorld


torch.set_num_threads(1)


def sample(dtype=torch.float32):
    torch.manual_seed(102)
    values = torch.randn(2, 23, 13, dtype=dtype)
    observed = torch.rand(2, 23, 13) > 0.25
    observed[:, -1, 0] = False
    observed[:, -1, 1] = True
    ages = torch.rand_like(values)
    clock = torch.randn(2, 23, 6, dtype=dtype).tanh()
    owners = torch.tensor([2, 9])
    hours = torch.arange(23)[None].expand(2, -1)
    return values, observed, ages, clock, owners, hours


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_parameter_budget_gradient_and_conditional_reconstruction(dtype):
    model = ReconstructionSensorWorld(floor_rate=torch.full((6,), 0.03)).to(dtype)
    inputs = sample(dtype)
    prediction, state, trace = model(*inputs, return_trace=True)
    assert sum(p.numel() for p in model.parameters() if p.requires_grad) == 98495
    assert sum(p.numel() for p in model.parameters()) == 98495
    assert sum(x.numel() for x in (state.source_sum, state.source_mass, state.evidence_sum, state.memory)) == 2 * 152

    missing = (~inputs[1][:, -1, :6]).to(dtype)
    safe_current = torch.where(inputs[1][:, -1], inputs[0][:, -1], torch.zeros_like(inputs[0][:, -1]))
    current_event = model.encoder(torch.cat((
        safe_current,
        inputs[1][:, -1].to(dtype),
        inputs[2][:, -1],
        inputs[3][:, -1],
        model.station(inputs[4]),
    ), -1)).tanh()
    write_key = F.normalize(model.address(current_event), dim=-1, eps=1e-6)
    belief_read = (state.memory @ write_key.unsqueeze(-1)).squeeze(-1)
    raw = model.current_head(torch.cat((
        8 * state.source_sum[:, :13] / state.source_mass[:, :13].clamp_min(1e-6),
        state.source_mass[:, :13],
        state.evidence_sum / state.count.clamp_min(1).to(dtype),
        inputs[2][:, -1, :6],
        belief_read,
    ), -1))
    anchor = torch.where(inputs[1][:, -1, :6], inputs[0][:, -1, :6],
                         8 * state.source_sum[:, :6] / state.source_mass[:, :6].clamp_min(1e-6))
    qmean = anchor + 2 * raw[:, :6].tanh()
    latent = trace['latent_state_variance']
    qvar = 0.0025 + F.softplus(raw[:, 6:]).square() + latent
    age_hours = torch.expm1(inputs[2][:, -1, :6] * math.log(257.))
    floor = missing * model.floor_rate * age_hours / 24
    expected_mean = trace['base_location'] + trace['affine_response'] * (qmean - anchor) * missing
    # Trace equations are exposed so tests can verify the declared conditional emission.
    torch.testing.assert_close(trace['current_anchor'], anchor)
    torch.testing.assert_close(trace['belief_read'], belief_read)
    torch.testing.assert_close(trace['shared_write_key'], write_key)
    torch.testing.assert_close(trace['current_mean'], qmean)
    torch.testing.assert_close(trace['current_variance'], qvar)
    torch.testing.assert_close(trace['floor_variance'], floor)
    expected_var = torch.maximum(
        trace['future_variance'] + trace['affine_response'].square() * qvar * missing,
        floor,
    )
    torch.testing.assert_close(trace['total_variance'], expected_var)
    torch.testing.assert_close(prediction.log_scale.square(), expected_var)
    torch.testing.assert_close(prediction.log_location, expected_mean)

    targets = torch.randn(2, 6, dtype=dtype)
    target_mask = torch.tensor([[True, False, True, False, True, False],
                                [False, True, False, True, False, True]])
    recon = model.reconstruction_loss(trace, targets, target_mask)
    expected_nll = 0.5 * ((targets - qmean).square() / qvar + qvar.log() + math.log(2 * math.pi))
    torch.testing.assert_close(recon, expected_nll[target_mask].mean())
    loss = prediction.log_location.square().mean() + prediction.log_scale.square().mean() + recon
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    assert all(p.grad.norm() > 0 for p in model.parameters())


def test_floor_is_monotone_and_missing_numeric_values_are_ignored():
    model = ReconstructionSensorWorld(floor_rate=torch.zeros(6)).double()
    inputs = list(sample(torch.float64))
    _, _, zero_floor = model(*inputs, return_trace=True)
    model.floor_rate.copy_(torch.full((6,), 0.4, dtype=torch.float64))
    _, _, positive_floor = model(*inputs, return_trace=True)
    missing = ~inputs[1][:, -1, :6]
    assert (positive_floor['total_variance'][missing] >= zero_floor['total_variance'][missing]).all()
    assert (positive_floor['floor_variance'][missing] >= zero_floor['floor_variance'][missing]).all()
    assert (positive_floor['floor_variance'][~missing] == 0).all()

    changed = list(inputs)
    changed[0] = inputs[0].clone()
    changed[0][~inputs[1]] = 7.9
    a, _, ta = model(*inputs, return_trace=True)
    b, _, tb = model(*changed, return_trace=True)
    torch.testing.assert_close(a.log_location, b.log_location, rtol=0, atol=0)
    torch.testing.assert_close(a.log_scale, b.log_scale, rtol=0, atol=0)
    torch.testing.assert_close(ta['current_mean'], tb['current_mean'], rtol=0, atol=0)


def test_chunked_forward_matches_single_call_and_validates_floor_rate():
    with pytest.raises(ValueError, match='floor_rate'):
        ReconstructionSensorWorld(floor_rate=torch.ones(5))
    model = ReconstructionSensorWorld(floor_rate=torch.full((6,), 0.02)).double()
    inputs = sample(torch.float64)
    whole, _, whole_trace = model(*inputs, return_trace=True)
    first, state = model(*(x[:, :10] if x.ndim >= 2 else x for x in inputs))
    chunked, _, chunk_trace = model(
        *(x[:, 10:] if x.ndim >= 2 else x for x in inputs), state=state, return_trace=True
    )
    torch.testing.assert_close(whole.log_location, chunked.log_location, rtol=1e-9, atol=1e-10)
    torch.testing.assert_close(whole.log_scale, chunked.log_scale, rtol=1e-9, atol=1e-10)
    for name in ('current_mean', 'current_variance', 'belief_read', 'floor_variance', 'total_variance'):
        torch.testing.assert_close(whole_trace[name], chunk_trace[name], rtol=1e-9, atol=1e-10)
