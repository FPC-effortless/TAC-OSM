"""LATFUSE002 constructed engineering guards; never world-model skill evidence."""
import math
import torch
from tac_osm.latent_fusion_002 import FusedLatentFixedState, state_contract
from tac_osm.latent_fusion_001 import FusedLatentCandidate


def test_exact_same_fp32_carried_state_and_active_100k_budget():
    old=FusedLatentCandidate(13,6,hidden=80,key_dim=16,val_dim=16,entity_count=16)
    new=FusedLatentFixedState()
    s=new.initial(2)
    o=old.initial(2)
    assert all(x.shape==y.shape for x,y in zip(s,o))
    assert all(x.dtype==y.dtype for x,y in zip(s,o))
    c=state_contract(new)
    assert c==dict(float_values=1696,occupancy_flags=16,nominal_fp32_bytes=6800,
                   dtype="torch.float32",trainable_parameters=98775,deployed_learned_parameters=98775)


def test_initial_retention_write_rates_are_explicit_and_not_random():
    model=FusedLatentFixedState()
    x=torch.randn(4,80)
    alpha=torch.sigmoid(model.retention_map(x))
    beta=torch.sigmoid(model.write_map(x))
    torch.testing.assert_close(alpha,torch.full((4,1),.995),atol=1e-6,rtol=0)
    torch.testing.assert_close(beta,torch.full((4,1),.1),atol=1e-6,rtol=0)
    assert model.retention_map.weight.requires_grad and model.write_map.weight.requires_grad


def test_multirate_slow_channels_preserve_longest_512_but_still_bounded():
    m=FusedLatentFixedState()
    r=m.slow_rates
    assert r.shape==(80,) and r.dtype==torch.float32
    assert .89<float(r[0])<.91 and float(r[-1])>.9999
    assert float(r[-1]**512)>.99
    assert bool((r>0).all() and (r<1).all())
    assert float((r[0]**512))<1e-10


def test_forecast_queries_belief_and_never_mutates_given_state():
    torch.manual_seed(8)
    m=FusedLatentFixedState()
    x=torch.randn(2,13)
    z=m.initial(2)
    state=m.step(z,x,torch.ones_like(x,dtype=torch.bool),
                 torch.zeros_like(x),torch.zeros_like(x),torch.tensor([0,1]))
    snapshots=[x.clone() for x in state]
    future=torch.randn(2,4,6)
    before=m.forecast(state,torch.tensor([0,1]),future)
    perm=[3,1,2,0]
    after=m.forecast(state,torch.tensor([0,1]),future[:,perm])
    for u,v in zip(before,after):
        torch.testing.assert_close(u[:,perm],v,rtol=0,atol=0)
    assert all(torch.equal(x,y) for x,y in zip(state,snapshots))


def test_autograd_loss_reaches_retention_query_and_encoder_without_extra_state():
    torch.manual_seed(9)
    m=FusedLatentFixedState()
    z=m.initial(2)
    for t in range(3):
        x=torch.randn(2,13)
        z=m.step(z,x,torch.ones_like(x,dtype=torch.bool),
                 torch.zeros_like(x),torch.zeros_like(x),torch.tensor([0,1]))
    mean,scale=m.forecast(z,torch.tensor([0,1]),torch.randn(2,1,6))
    loss=(mean**2+scale).sum()
    loss.backward()
    for name in ("retention_map","write_map","belief_query","key_map","embed"):
        assert all(p.grad is not None and torch.isfinite(p.grad).all()
                   for p in getattr(m,name).parameters()),name


def test_original_source_state_calculus_unchanged():
    model=FusedLatentFixedState()
    s=model.initial(1)
    assert s.fast.numel()==80 and s.slow.numel()==80
    assert s.associative.numel()==16*16 and s.entities.numel()==16*80
    assert not any(x.dtype!=torch.float32 for x in s[:4])
