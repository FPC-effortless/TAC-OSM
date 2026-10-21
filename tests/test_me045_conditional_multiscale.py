"""ME045 numerical-only and exact hard conditional native compute contracts."""
import torch
from tac_osm.me044_multiscale_decoder import build_model as old_model
from tac_osm.me045_conditional_multiscale import ARMS,make_model


def fresh(arm,h=96,period=24):
    torch.set_num_threads(1)
    cheap=old_model("original_context",h,45,period)
    return make_model(arm,h,45,period,cheap),cheap


def test_me045_all_three_new_full_model_backbones_are_exactly_frozen():
    x=torch.randn(2,96,3)
    for arm in ARMS:
        net,cheap=fresh(arm)
        net.eval();cheap.eval()
        with torch.no_grad():
            assert torch.equal(net.routed(x,"cheap")[0],cheap(x))
            assert torch.equal(net.routed(x,"always")[0],cheap(x))
        assert net.active_parameters()>0
        assert all(not p.requires_grad for n,p in net.named_parameters()
                   if not n.startswith(("branch.","branch_projection.",
                                        "coefficients.","trust.","router.")))


def test_me045_hard_skip_no_expert_invocation_and_real_execution():
    x=torch.randn(1,96,3)
    for arm in ARMS:
        net,_=fresh(arm)
        net.eval()
        calls=[]
        hook=net.branch.register_forward_pre_hook(lambda *_:calls.append(1))
        with torch.no_grad():
            net.routed(x,"cheap")
            assert calls==[]
            net.router[-1].weight.zero_()
            net.router[-1].bias.fill_(-20.)
            out,base,p,chosen=net.routed(x,"adaptive")
            assert calls==[] and not chosen.any()
            assert torch.equal(out,base)
            net.router[-1].bias.fill_(20.)
            out,base,p,chosen=net.routed(x,"adaptive")
            assert len(calls)==1 and chosen.all()
            assert p.shape==(3,1)
        hook.remove()


def test_me045_true_FIT_loss_updates_expert_and_router_without_cheap_grads():
    x=torch.randn(2,96,3)
    y=torch.randn(2,96,3)
    for arm in ARMS:
        net,_=fresh(arm)
        optimizer=torch.optim.AdamW([p for p in net.parameters() if p.requires_grad],lr=.003)
        for step in range(3):
            optimizer.zero_grad(set_to_none=True)
            loss,expert,router=net.expert_router_loss(x,y)
            assert torch.isfinite(loss) and torch.isfinite(expert) and torch.isfinite(router)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in net.parameters() if p.requires_grad],1.)
            optimizer.step()
        assert all(p.grad is None for n,p in net.named_parameters() if not p.requires_grad)
        assert any(p.grad is not None and p.grad.abs().sum()>0
                   for n,p in net.named_parameters() if n.startswith("router."))
        assert any(p.grad is not None and p.grad.abs().sum()>0
                   for n,p in net.named_parameters() if n.startswith("branch."))
        assert any(p.grad is not None and p.grad.abs().sum()>0
                   for n,p in net.named_parameters() if n.startswith("coefficients."))


def test_me045_native_and_gru40_close_parameter_counts_but_gru16_unmatched():
    for h in (96,720):
        native,_=fresh("conditional_native",h)
        matched,_=fresh("conditional_gru40",h)
        small,_=fresh("conditional_gru16",h)
        n= sum(p.numel() for p in native.parameters())
        g= sum(p.numel() for p in matched.parameters())
        s= sum(p.numel() for p in small.parameters())
        assert abs(n-g)/max(n,g)<.01
        assert s<n and s<g
        assert native.active_parameters()>small.active_parameters()


def test_me045_bounded_full_residual_and_no_unseen_future_inputs():
    x=torch.randn(2,96,3)
    net,_=fresh("conditional_native")
    with torch.no_grad():
        net.coefficients[-1].bias.fill_(100.)
        pred,cheap,q,chosen=net.routed(x,"always")
        scale=torch.sqrt((x-x.mean(1,keepdim=True)).square().mean(1,keepdim=True)+1e-5)
        assert (pred-cheap).abs().le(.5*scale+1e-6).all()
        assert chosen.all()
        assert q.shape==(6,1)
