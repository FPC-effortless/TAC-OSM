"""Mathematical/causal tests, never synthetic capability evidence."""
import torch
from tac_osm.coupled_native import CoupledNative, joint_jacobian, rms_jacobian
from scripts.me019_authentic_history import NativeExtended


def values():
    torch.manual_seed(24024)
    meta = torch.rand(3,128,3)
    meta[:,:,1] = (meta[:,:,1]>.5).float()
    return torch.randint(16000,(3,128)), meta, torch.randint(16000,(3,)), torch.rand(3,1)


def modes():
    return [(a, r) for a in (0., .25) for r in (False, True)]


def test_plain_matches_original_weights_and_nonzero_head_forecasts():
    torch.set_num_threads(1)
    torch.manual_seed(93); old = NativeExtended(128)
    torch.manual_seed(93); new = CoupledNative(0., False, 0., 1.)
    for name, weight in old.state_dict().items():
        assert torch.equal(weight, new.state_dict()[name])
    with torch.no_grad():
        old.head.weight.normal_(std=.2); new.head.load_state_dict(old.head.state_dict())
    torch.testing.assert_close(old(*values()), new(*values()), atol=1e-6, rtol=0)
    assert sum(p.numel() for p in new.parameters()) == 135050


def test_full_coupled_jacobian_matches_all_autograd_entries():
    torch.manual_seed(94)
    u, h = torch.randn(11, dtype=torch.float64), torch.rand(64, dtype=torch.float64)*2-1
    for alpha, rms in modes():
        model = CoupledNative(alpha, rms, .7, .05).double()
        j, a, d, b, c = joint_jacobian(model, u, h)
        auto = torch.autograd.functional.jacobian(lambda x: model.advance_features(u[None], x[None])[0], h)
        torch.testing.assert_close(j, auto, atol=1e-12, rtol=1e-11)
        if alpha == 0.: assert d.abs().max() == 0
        else: assert d.abs().sum() > 0


def test_joint_jacobian_factorization_and_determinant():
    model = CoupledNative(.25, False, 0., 1.).double()
    j, a, d, b, c = joint_jacobian(model, torch.randn(11, dtype=torch.float64), torch.zeros(64, dtype=torch.float64))
    eye, zero = torch.eye(32, dtype=torch.float64), torch.zeros(32,32,dtype=torch.float64)
    left = torch.cat((torch.cat((eye,zero),1),torch.cat((b,eye),1)),0)
    right = torch.cat((torch.cat((a,d),1),torch.cat((zero,c),1)),0)
    torch.testing.assert_close(j, left@right, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(torch.linalg.det(j), torch.linalg.det(a)*torch.linalg.det(c), atol=1e-12, rtol=1e-10)


def test_split_rms_actual_derivative_and_zero_state_bound():
    model = CoupledNative(0., True, 0., 1.).double()
    x = torch.randn(64, dtype=torch.float64)
    actual = torch.autograd.functional.jacobian(lambda h:model.state_features(h[None])[0],x)
    expected = torch.block_diag(rms_jacobian(x[:32]),rms_jacobian(x[32:]))
    torch.testing.assert_close(actual,expected,atol=1e-12,rtol=1e-12)
    assert torch.linalg.matrix_norm(actual,2) <= 10
    zero = torch.zeros(32,dtype=torch.float64)
    torch.testing.assert_close(rms_jacobian(zero),10*torch.eye(32,dtype=torch.float64))
    assert model.state_features(x[None]).norm() <= 8


def test_prefix_prediction_is_causal_in_all_cells():
    args = values()
    for alpha, rms in modes():
        net = CoupledNative(alpha, rms, .7, .05)
        with torch.no_grad(): net.head.weight.normal_(std=.1)
        out = net.prefix_logits(*args)
        for col, depth in enumerate((32,64,96)):
            m = args[1].clone(); m[:,depth:,1] = 1-m[:,depth:,1]
            changed = net.prefix_logits(args[0],m,args[2],args[3])
            torch.testing.assert_close(out[:,col],changed[:,col],atol=0,rtol=0)


def test_streaming_prefix_and_window_match_with_bounded_states():
    args = values()
    for alpha, rms in modes():
        net = torch.jit.script(CoupledNative(alpha,rms,.7,.05))
        state = torch.zeros(3,64)
        with torch.no_grad():
            for t in range(128):
                state = net.stream_step(args[0][:,t],args[1][:,t],state)
                assert state.abs().max() <= 1
            torch.testing.assert_close(state,net.encode(args[0],args[1]),atol=1e-6,rtol=0)
            torch.testing.assert_close(net.prefix_logits(*args)[:,-1:],net(*args),atol=0,rtol=0)


def test_feature_mean_preserves_latent_logit_level():
    for alpha, rms in modes():
        net = CoupledNative(alpha,rms,0.,1.)
        with torch.no_grad(): net.head.weight.normal_()
        h = torch.rand(16,64)*2-1
        features = net.state_features(h)
        out = net.readout_features(features,torch.zeros(16,dtype=torch.long),torch.zeros(16,1),torch.full((16,1),.5))
        reference = net.readout_features(features.mean(0,keepdim=True),torch.zeros(1,dtype=torch.long),torch.zeros(1,1),torch.full((1,1),.5))
        torch.testing.assert_close(out.mean(),reference.squeeze(),atol=1e-6,rtol=1e-6)


def test_active_gradients_optimizer_and_standalone_roundtrip(tmp_path):
    args = values()
    for alpha, rms in modes():
        net = torch.jit.script(CoupledNative(alpha,rms,.7,.05))
        with torch.no_grad(): net.head.weight.normal_(std=.1)
        opt = torch.optim.AdamW(net.parameters(),lr=.001)
        net.prefix_logits(*args).square().mean().backward()
        for name,p in net.named_parameters():
            assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum()>0,name
        opt.step()
        path=tmp_path/'net.pt'; torch.jit.save(net.eval(),str(path))
        restored=torch.jit.load(str(path))
        with torch.no_grad(): torch.testing.assert_close(net(*args),restored(*args),atol=0,rtol=0)
