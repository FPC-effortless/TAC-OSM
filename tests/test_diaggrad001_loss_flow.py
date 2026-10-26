"""Synthetic arithmetic checks only; no real model optimizer steps."""
import torch

from scripts.diaggrad001_loss_flow import block_name, geometry, shadow_adamw
from tac_osm.variants.point_priority_sensor_world import PointPrioritySensorWorld


def test_parameter_partition_is_complete():
    model = PointPrioritySensorWorld()
    assert sum(p.numel() for p in model.parameters()) == 100349
    assert all(block_name(name) for name, _ in model.named_parameters())


def test_geometry_zero_vectors_are_not_fabricated_cosines():
    assert geometry([torch.zeros(2)], [torch.ones(2)])['cosine'] is None
    assert geometry([torch.ones(2)], [-torch.ones(2)])['cosine'] == -1.


def test_shadow_adamw_matches_torch_without_mutating_inputs():
    # Two toy parameters, not a TAC fit. Exercise moments, rates, clipping, decay.
    p = torch.nn.Parameter(torch.tensor([.4, -.8], dtype=torch.float64))
    q = torch.nn.Parameter(torch.tensor([.2], dtype=torch.float64))
    opt = torch.optim.AdamW([{'params': [p], 'lr': 3e-6},
                             {'params': [q], 'lr': 3e-4}], weight_decay=.001)
    p.grad, q.grad = torch.tensor([.2, -.4], dtype=torch.float64), torch.tensor([.3], dtype=torch.float64)
    opt.step()
    named = [('encoder.weight', p), ('median_head.weight', q)]
    before = [x.detach().clone() for x in (p, q)]
    grads = [torch.tensor([2., 3.], dtype=torch.float64), torch.tensor([-4.], dtype=torch.float64)]
    delta, _, clip = shadow_adamw([p, q], grads, {'optimizer': opt.state_dict()}, named)
    assert all(torch.equal(a, b) for a, b in zip(before, (p, q)))
    p.grad, q.grad = grads
    torch.nn.utils.clip_grad_norm_([p, q], 1.)
    opt.step()
    for old, current, expected in zip(before, (p, q), delta):
        torch.testing.assert_close(current-old, expected, rtol=1e-9, atol=1e-16)
    assert clip < 1.
