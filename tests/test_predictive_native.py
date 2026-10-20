"""Mathematical/causal falsifiers only. Constructed inputs imply no capability."""
import math
import torch
from tac_osm.predictive_native import PredictiveNative
from scripts.me019_authentic_history import NativeExtended


def values():
    torch.manual_seed(23023)
    meta = torch.rand(3, 128, 3)
    meta[:, :, 1] = (meta[:, :, 1] > .5).float()
    return (torch.randint(16000, (3, 128)), meta,
            torch.randint(16000, (3,)), torch.rand(3, 1))


def test_seeded_native_identity_and_zero_interaction():
    torch.set_num_threads(1)
    torch.manual_seed(91)
    old = NativeExtended(128)
    torch.manual_seed(91)
    new = PredictiveNative(True, 0., 1.)
    for name, weight in old.state_dict().items():
        assert torch.equal(weight, new.state_dict()[name])
    assert sum(p.numel() for p in new.parameters()) == 135562
    args = values()
    torch.testing.assert_close(old(*args), new(*args), atol=0, rtol=0)


def test_prefix_and_script_final_are_the_same_native_forecast():
    args = values()
    for conditional in (False, True):
        net = PredictiveNative(conditional, .7, .05)
        with torch.no_grad():
            net.head.weight.normal_(std=.1)
            net.interaction.normal_(std=.2)
        script = torch.jit.script(net)
        torch.testing.assert_close(net(*args), script(*args), atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(script.prefix_logits(*args)[:, -1:], script(*args), atol=0, rtol=0)


def test_prefix_never_reads_its_target_feedback_or_later_feedback():
    args = values()
    for conditional in (False, True):
        net = PredictiveNative(conditional, .7, .05)
        with torch.no_grad():
            net.head.weight.normal_(std=.2)
            net.interaction.normal_(std=.3)
        original = net.prefix_logits(*args)
        for col, depth in enumerate((32, 64, 96)):
            changed = args[1].clone()
            changed[:, depth:, 1] = 1.-changed[:, depth:, 1]
            new = net.prefix_logits(args[0], changed, args[2], args[3])
            torch.testing.assert_close(new[:, col], original[:, col], atol=0, rtol=0)


def test_added_readout_bound_and_exact_mixed_derivative():
    torch.manual_seed(104)
    u = torch.randn(64, 8, dtype=torch.float64)
    b = 4*torch.tanh(u)/math.sqrt(512)
    h = torch.rand(64, dtype=torch.float64)*2-1
    e = torch.randn(8, dtype=torch.float64, requires_grad=True)
    def gh(query): return b @ (torch.tanh(query)/math.sqrt(8))/8
    mixed = torch.autograd.functional.jacobian(gh, e)
    expected = b @ torch.diag(1-torch.tanh(e).square())/(8*math.sqrt(8))
    torch.testing.assert_close(mixed, expected, atol=1e-12, rtol=1e-12)
    net = PredictiveNative(True, 0., 1.).double()
    with torch.no_grad():
        net.interaction.copy_(u)
        net.emb.weight[0].copy_(e)
    actual_h = h.clone().requires_grad_(True)
    out = net.readout(actual_h[None], torch.tensor([0]),
                      torch.zeros(1, 1, dtype=torch.float64),
                      torch.full((1, 1), .5, dtype=torch.float64))
    actual_grad = torch.autograd.grad(out.sum(), actual_h, create_graph=True)[0]
    torch.testing.assert_close(actual_grad, gh(e), atol=1e-12, rtol=1e-12)
    actual_mixed_sum = torch.autograd.grad(actual_grad.sum(), net.emb.weight)[0][0]
    torch.testing.assert_close(actual_mixed_sum, expected.sum(0), atol=1e-12, rtol=1e-12)
    assert gh(e).norm() <= .5
    assert abs((h/8) @ b @ (torch.tanh(e)/math.sqrt(8))) <= 4
    saturated = torch.full((64, 8), 4/math.sqrt(512), dtype=torch.float64)
    torch.testing.assert_close((torch.ones(64, dtype=torch.float64)/8) @ saturated @
                               (torch.ones(8, dtype=torch.float64)/math.sqrt(8)),
                               torch.tensor(4., dtype=torch.float64))


def test_action_query_can_express_nonadditive_log_odds():
    state = torch.zeros(4, 64)
    state[:2, 0] = 1
    state[2:, 0] = -1
    actions = torch.tensor([0, 1, 0, 1])
    for conditional in (False, True):
        net = PredictiveNative(conditional, 0., 1.)
        with torch.no_grad():
            net.emb.weight[:2].zero_()
            net.emb.weight[0, 0], net.emb.weight[1, 0] = 1., -1.
            net.interaction[0, 0] = 1.
        out = net.readout(state, actions, torch.zeros(4, 1), torch.full((4, 1), .5)).flatten()
        delta = out[0]-out[1]-out[2]+out[3]
        assert (float(delta.detach()) > .01) if conditional else float(delta.detach()) == 0


def test_all_parameter_tensors_have_finite_active_gradient_paths():
    args = values()
    for conditional in (False, True):
        net = PredictiveNative(conditional, .7, .05)
        with torch.no_grad():
            net.head.weight.normal_(std=.1)
            net.interaction.normal_(std=.2)
        net.prefix_logits(*args).square().mean().backward()
        for name, parameter in net.named_parameters():
            assert parameter.grad is not None, name
            assert torch.isfinite(parameter.grad).all(), name
            assert parameter.grad.abs().sum() > 0, name


def test_scripted_optimizer_checkpoint_and_standalone_roundtrip(tmp_path):
    args = values()
    model = torch.jit.script(PredictiveNative(True, .7, .05))
    opt = torch.optim.AdamW(model.parameters(), lr=.001)
    model.prefix_logits(*args).square().mean().backward()
    opt.step()
    path = tmp_path/"weights.pt"
    torch.save({"weights": model.state_dict(), "optimizer": opt.state_dict()}, path)
    model.load_state_dict(torch.load(path, weights_only=True)["weights"])
    torch.jit.save(model.eval(), str(tmp_path/"model.jit.pt"))
    restored = torch.jit.load(str(tmp_path/"model.jit.pt")).eval()
    with torch.no_grad():
        torch.testing.assert_close(model(*args), restored(*args), atol=0, rtol=0)
        assert model.bounded_matrix().norm() <= 4
