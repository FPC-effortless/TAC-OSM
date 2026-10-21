"""ME046 exact numerical safety and joint-training *software* checks.

Constructed tensors verify equations only; they are never real-data capability
or model forecasting success evidence.
"""
import torch
from tac_osm.me044_multiscale_decoder import build_model as original_build
from tac_osm.me046_joint_multiscale import (
    ARMS, NATIVE_ARMS, LOOP_ARMS, build_model, objective, preflight
)


def test_me046_six_complete_recurrent_and_nonrecurrent_architectures():
    result=preflight()
    assert result["status"]=="ME046_NUMERICAL_MATH_ONLY_PASS"
    assert len(result["arms"])==6
    assert result["weights_tied_refinement_steps"]==2
    assert not result["real_model_scores_computed"]


def test_me046_original_pretrained_real_context_weights_preserved_exactly():
    torch.set_num_threads(1)
    torch.manual_seed(461)
    x=torch.randn(2,96,3)
    old=original_build("original_context",96,46001,24)
    for arm in ARMS:
        new=build_model(arm,96,46001,24,old)
        cheap,pred,*_=new.full_parts(x)
        assert torch.equal(cheap,old(x)),arm
        assert torch.equal(pred,old(x)),arm
        assert all(p.requires_grad for p in new.parameters())


def test_me046_tied_residual_is_bounded_and_conservative():
    torch.set_num_threads(1)
    torch.manual_seed(461)
    for arm in LOOP_ARMS:
        model=build_model(arm,96,46001,24)
        z=torch.randn(4,64)
        r=model.refiner(z)
        alpha=torch.sigmoid(model.refiner.eta)
        assert torch.isfinite(r).all()
        assert (r-z).abs().max().item()<=alpha.item()+1e-6
        assert len([n for n,p in model.refiner.named_parameters() if n.startswith("map.")])==2


def test_me046_joint_objective_trains_context_and_expert_not_frozen():
    torch.set_num_threads(1)
    torch.manual_seed(461)
    x=torch.randn(2,96,3)
    y=torch.randn(2,96,3)
    for arm in ARMS:
        n=build_model(arm,96,46001,24)
        opt=torch.optim.AdamW(n.parameters(),lr=.001)
        for _ in range(2):
            opt.zero_grad(set_to_none=True)
            loss,full,cheap=objective(n,x,y)
            assert torch.isfinite(loss) and torch.isfinite(full) and torch.isfinite(cheap)
            loss.backward()
            opt.step()
        assert n.ar.weight.grad is not None if hasattr(n.ar,"weight") else True
        assert any(p.grad is not None and p.grad.abs().sum()>0
                   for name,p in n.named_parameters() if name.startswith("encoder."))
        assert any(p.grad is not None and p.grad.abs().sum()>0
                   for name,p in n.named_parameters() if name.startswith("coefficients."))
