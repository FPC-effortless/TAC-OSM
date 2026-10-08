import torch

from tac_osm.generic_casm_001 import GenericCASM001


def test_forward_shapes():
    model = GenericCASM001()
    support_u = torch.randn(5, 8, 8)
    support_v = torch.randn(5, 8, 8)
    support_y = torch.randn(5, 8, 8)
    query_u = torch.randn(5, 4, 8)
    query_v = torch.randn(5, 4, 8)
    out = model(support_u, support_v, support_y, query_u, query_v)
    assert out.shape == (5, 4, 8)


def test_support_row_permutation_invariant_at_forward_value():
    torch.manual_seed(0)
    model = GenericCASM001()
    su = torch.randn(2, 8, 8)
    sv = torch.randn(2, 8, 8)
    sy = torch.randn(2, 8, 8)
    qu = torch.randn(2, 4, 8)
    qv = torch.randn(2, 4, 8)
    a = model(su, sv, sy, qu, qv)
    p = torch.tensor([7, 1, 5, 0, 3, 6, 2, 4])
    b = model(su[:, p], sv[:, p], sy[:, p], qu, qv)
    assert torch.allclose(a, b, atol=1e-6)


def test_gradients_reach_operator_code_and_execution_basis():
    torch.manual_seed(1)
    model = GenericCASM001()
    su = torch.randn(4, 8, 8)
    sv = torch.randn(4, 8, 8)
    sy = torch.randn(4, 8, 8)
    qu = torch.randn(4, 4, 8)
    qv = torch.randn(4, 4, 8)
    y = torch.randn(4, 4, 8)
    loss = torch.nn.functional.mse_loss(model(su, sv, sy, qu, qv), y)
    loss.backward()
    assert model.support_encoder[0].weight.grad is not None
    assert model.code_head[3].weight.grad is not None
    assert model.u_proj.grad is not None
    assert model.v_proj.grad is not None
    assert model.out_proj.grad is not None
