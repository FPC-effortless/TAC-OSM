"""Probability semantics and frozen-body checks, independent of trained fits."""
import pytest
import copy
from pathlib import Path
import sys
torch = pytest.importorskip('torch')
from tac_osm.development.semantic_value import ConditionalExpectedValue, from_parent
from tac_osm.development.public_outcome import PublicOutcomeAgent
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from run_semantic_value_dev069 import save_training_state, restore_training_state


def inputs():
    probabilities = torch.randn(3, 5, 21, dtype=torch.float64).softmax(-1)
    return torch.cat((probabilities, torch.rand(3, 5, 1, dtype=torch.float64),
                      torch.randn(3, 5, 8, dtype=torch.float64)), -1)


def test_policy_is_affine_in_probabilities_at_fixed_public_context():
    policy = ConditionalExpectedValue().double(); a = inputs(); b = a.clone()
    b[..., :21] = torch.randn(3, 5, 21, dtype=torch.float64).softmax(-1)
    weight = .37
    mixed = weight*a+(1-weight)*b
    torch.testing.assert_close(policy(mixed), weight*policy(a)+(1-weight)*policy(b), atol=1e-12, rtol=0)


def test_more_predicted_cost_cannot_improve_value_at_fixed_outcomes():
    policy = ConditionalExpectedValue().double(); x = inputs().requires_grad_()
    gradient = torch.autograd.grad(policy(x).sum(), x)[0][..., 21]
    assert torch.isfinite(gradient).all() and (gradient > 0).all()
    higher = x.detach().clone(); higher[..., 21] += .25
    assert (policy(higher) > policy(x.detach())).all()
    with pytest.raises(ValueError): policy(torch.zeros(3, 5, 29, dtype=torch.float64))


def test_policy_replacement_keeps_every_world_weight_and_prediction():
    parent = PublicOutcomeAgent(seed=12).double().eval()
    spec = dict(backend='tac', width=32, aux=32)
    model = from_parent(parent.state_dict(), spec, 12, 13, 'expected').eval()
    events = torch.randn(2, 8, 46, dtype=torch.float64)
    present = torch.ones(2, 8, dtype=torch.bool)
    keys = torch.randn(2, 2, 16, dtype=torch.float64)
    candidates = torch.randn(2, 5, 19, dtype=torch.float64)
    prices = torch.ones(2, 2, dtype=torch.float64); remaining = torch.ones(2, 1, dtype=torch.float64)
    a = parent(events, present, keys, candidates, prices, remaining)
    b = model(events, present, keys, candidates, prices, remaining)
    for name in ('posterior', 'future', 'cost', 'features'): torch.testing.assert_close(a[name], b[name], atol=0, rtol=0)
    assert all(name.startswith('policy.') for name, p in model.named_parameters() if p.requires_grad)


def test_parameter_budget_and_serialization_match():
    parent = PublicOutcomeAgent().double(); spec = dict(backend='tac', width=32, aux=32)
    models = {r: from_parent(parent.state_dict(), spec, 0, 99, r) for r in ('mlp', 'expected')}
    counts = {r: sum(p.numel() for p in m.policy.parameters()) for r, m in models.items()}
    assert counts == dict(mlp=6209, expected=6166)
    assert abs(counts['expected']/counts['mlp']-1) < .01
    clone = from_parent(parent.state_dict(), spec, 0, 99, 'expected')
    clone.load_state_dict(models['expected'].state_dict())
    x = inputs()
    torch.testing.assert_close(models['expected'].policy(x), clone.policy(x), atol=0, rtol=0)


def test_actual_checkpoint_restores_optimizer_rng_and_ema_exactly(tmp_path):
    torch.manual_seed(77); initial = ConditionalExpectedValue().double()
    x = inputs().reshape(15, 30); y = torch.rand(15, 1, dtype=torch.float64)
    def create():
        raw = {'arm':copy.deepcopy(initial)}; ema = {'arm':copy.deepcopy(initial).requires_grad_(False)}
        opt = {'arm':torch.optim.AdamW(raw['arm'].parameters(), lr=.003)}
        generator = torch.Generator().manual_seed(3)
        return raw, ema, opt, generator
    def step(raw, ema, opt, generator):
        ids = torch.randint(15, (8,), generator=generator)
        loss = (raw['arm'](x[ids])-y[ids]).square().mean()
        opt['arm'].zero_grad(); loss.backward(); opt['arm'].step()
        with torch.no_grad():
            for a, b in zip(ema['arm'].parameters(), raw['arm'].parameters(), strict=True): a.mul_(.995).add_(b, alpha=.005)
    full = create()
    for _ in range(6): step(*full)
    partial = create()
    for _ in range(3): step(*partial)
    path = tmp_path/'training-state.pt'
    save_training_state(path, *partial, {'arm':[]}, 3, 'frozen-source', {'arm':.003}, 1.)
    restored = create()
    state = restore_training_state(path, *restored, 'frozen-source', {'arm':.003})
    assert state['update'] == 3 and not path.with_suffix('.pt.tmp').exists()
    for _ in range(3): step(*restored)
    for expected, actual in zip(full[:2], restored[:2], strict=True):
        for name, tensor in expected['arm'].state_dict().items():
            torch.testing.assert_close(tensor, actual['arm'].state_dict()[name], atol=0, rtol=0)
    assert torch.equal(full[3].get_state(), restored[3].get_state())
    with pytest.raises(AssertionError): restore_training_state(path, *create(), 'changed-source', {'arm':.003})
