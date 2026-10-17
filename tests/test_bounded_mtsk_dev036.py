"""Meaningful parameter, stability and addressing contracts for DEV-036."""
import torch
from tac_osm.development.bounded_mtsk import build_bounded
from tac_osm.development.local_mtsk import LocalMTSKPredictor
from tac_osm.development.noisy_memory import parameter_hash


def test_identical_trainable_initial_tensors_and_state_budget():
    control=LocalMTSKPredictor('mtsk_local_multi',34311)
    bounded=build_bounded()
    assert parameter_hash(control)==parameter_hash(bounded)
    assert sum(p.numel() for p in control.parameters())==sum(p.numel() for p in bounded.parameters())
    assert control.state_scalars==bounded.state_scalars==768


def test_conditional_row_operator_is_a_contraction_even_for_large_keys_and_beta():
    kernel=build_bounded().memory
    with torch.no_grad():
        kernel.key.weight.mul_(100)
        kernel.beta_logits.fill_(50.)
    packed=kernel.initial(torch.zeros(2,12))[:,0]
    x=torch.randn(2,12)
    _,alpha,beta,key,responsibility=kernel.transition(packed,x)
    assert (torch.linalg.vector_norm(key,dim=-1)<=1+1e-6).all()
    row_alpha=alpha.squeeze(-1).double()
    feedback=beta.squeeze(-1).double()*responsibility.double()*key.double().square().sum(-1).unsqueeze(-1).unsqueeze(-1)
    assert (feedback>=0).all() and (feedback<=row_alpha+1e-12).all()
    gain=torch.maximum(row_alpha.abs(),(row_alpha-feedback).abs())
    assert (gain<1.).all()
    # Perturb one value-coordinate row while holding its routing responsibility fixed.
    a=float(row_alpha[0,0,0].detach());b=float((beta[0,0,0,0]*responsibility[0,0,0]).detach());k=key[0].detach().double()
    matrix=a*torch.eye(3,dtype=torch.float64)-b*torch.outer(k,k)
    assert torch.allclose(torch.linalg.svdvals(matrix).max(),gain[0,0,0],atol=1e-7)


def test_local_transition_preserves_other_banks_and_gradients_exist():
    model=build_bounded();model.train();memory=model.memory
    initial=memory.initial(torch.zeros(2,12));entities=torch.tensor([1,5]);x=torch.randn(2,12)
    updated=memory.update(initial,x,entities)
    for partner,bank in enumerate(entities):
        other=torch.arange(8)!=bank
        assert torch.equal(initial[partner,other],updated[partner,other])
    loss=updated.square().sum();loss.backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
