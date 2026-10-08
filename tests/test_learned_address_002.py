import torch

from tac_osm.learned_address_002 import LearnedAddressMetric002


def test_identity_initialisation_matches_raw_dot():
    model = LearnedAddressMetric002()
    q = torch.randn(4, 16)
    k = torch.randn(4, 7, 16)
    got = model(q, k)
    expected = torch.einsum("bd,bmd->bm", q, k)
    assert torch.allclose(got, expected)


def test_metric_has_trainable_bilinear_and_candidate_bias():
    model = LearnedAddressMetric002()
    params = dict(model.named_parameters())
    assert "query.weight" in params
    assert "key.weight" in params
    assert "candidate_bias.0.weight" in params
    assert "candidate_bias.2.weight" in params
    assert all(p.requires_grad for p in model.parameters())


def test_metric_backpropagates_to_query_and_key():
    torch.manual_seed(0)
    model = LearnedAddressMetric002()
    q = torch.randn(8, 16)
    k = torch.randn(8, 5, 16)
    target = torch.randint(5, (8,))
    loss = torch.nn.functional.cross_entropy(model(q, k), target)
    loss.backward()
    assert model.query.weight.grad is not None
    assert model.key.weight.grad is not None
    assert model.candidate_bias[0].weight.grad is not None
