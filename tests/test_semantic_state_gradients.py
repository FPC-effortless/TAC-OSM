"""Verify the causal gradient and deployment boundaries of DEV065."""
import pytest
import torch
from tac_osm.development.belief_bottleneck import BeliefBottleneck
from tac_osm.development.semantic_state import SemanticStateAgent, configurations
from test_semantic_bottleneck import payload


def test_policy_gradient_isolation_does_not_block_proper_belief_learning():
    args = payload()
    model = SemanticStateAgent(seed=19).double()
    coupled = model.training_forward(*args)
    isolated = model.training_forward(*args, isolate=True)
    for name in ('posterior', 'logits', 'future', 'cost'):
        torch.testing.assert_close(coupled[name], isolated[name], atol=0, rtol=0)
    grad = torch.autograd.grad(coupled['logits'].square().sum(), coupled['posterior'], retain_graph=True)[0]
    assert grad.abs().sum() > 0
    assert torch.autograd.grad(isolated['logits'].square().sum(), isolated['posterior'], allow_unused=True, retain_graph=True)[0] is None
    target = torch.zeros_like(isolated['posterior']); target[:, 0] = 1
    loss = -(target * isolated['posterior'].log()).sum()
    assert torch.autograd.grad(loss, model.belief[0].weight)[0].abs().sum() > 0


def test_exact_training_targets_only_replace_explicit_rows_and_never_deployment():
    args = payload(); model = SemanticStateAgent(seed=23).double()
    normal = model(*args)
    target = torch.zeros(2, 12, dtype=torch.float64); target[:, 0] = 1
    mixed = model.training_forward(*args, semantic_target=target, exact_mask=torch.tensor([True, False]))
    expected = model.act_from_belief(target, args[2], args[3], args[4])
    for name in ('logits', 'future', 'cost'):
        torch.testing.assert_close(mixed[name][0], expected[name][0], atol=0, rtol=0)
        torch.testing.assert_close(mixed[name][1], normal[name][1], atol=0, rtol=0)
        torch.testing.assert_close(model(*args)[name], normal[name], atol=0, rtol=0)
    torch.testing.assert_close(mixed['posterior'], normal['posterior'], atol=0, rtol=0)
    with pytest.raises(ValueError):
        model.training_forward(*args, semantic_target=target)


@pytest.mark.parametrize('arm', ['tac', 'gru', 'no_casm', 'mlp'])
def test_new_encoder_preserves_original_inference_before_new_training(arm):
    args = payload(); old = BeliefBottleneck(arm=arm, seed=11).double()
    new = SemanticStateAgent(arm=arm, seed=11).double(); new.load_state_dict(old.state_dict())
    a, b = old(*args), new(*args)
    for key in a:
        torch.testing.assert_close(a[key], b[key], atol=0, rtol=0)


def test_semantic_interface_excludes_latent_and_matches_parameter_budgets():
    args = payload(); model = SemanticStateAgent(seed=31).double()
    output = model(*args)
    assert torch.autograd.grad(output['logits'].sum(), output['latent'], allow_unused=True, retain_graph=True)[0] is not None
    # Dependence on the latent is solely via q: externally supplied q has no history input.
    q = output['posterior'].detach().requires_grad_()
    actor = model.act_from_belief(q, args[2], args[3], args[4])
    assert torch.autograd.grad(actor['logits'].sum(), output['latent'], allow_unused=True)[0] is None
    specs = configurations(); target = specs['tac_isolated_mixed']['parameters']
    assert len(specs) == 6 and all(abs(s['parameters']/target-1) <= .05 for s in specs.values())
