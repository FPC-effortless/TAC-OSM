"""Complete ESWAM003 original/optimized parameter, forecast and gradient parity."""
import pytest
import torch
from tac_osm.variants.selective_exchange_world import SelectiveExchangeWorldModel
from tac_osm.variants.vectorized_exchange_world import VectorizedSelectiveExchangeWorldModel


def same(x,y):
    if x.dtype == torch.bool or not x.dtype.is_floating_point:
        assert torch.equal(x,y)
    else:
        torch.testing.assert_close(x,y,atol=5e-6,rtol=2e-5)


@pytest.mark.parametrize("batch", [1, 4])
def test_preferred_eswam003_complete_forward_backward_and_roles(batch):
    torch.manual_seed(31000 + batch)
    a=SelectiveExchangeWorldModel()
    b=VectorizedSelectiveExchangeWorldModel()
    b.load_state_dict(a.state_dict(),strict=True)
    assert sum(p.numel() for p in a.parameters())==sum(p.numel() for p in b.parameters())
    sa=a.initial_selective(batch);sb=b.initial_selective(batch)
    la,lb=0.0,0.0
    for t in range(5):
        obs=torch.randn(batch,14)
        mask=torch.rand(batch,14)>.15
        mask[0,:]=t!=2
        action=torch.randn(batch,14)
        dt=torch.full((batch,),.1)
        identity=torch.full((batch,),t//2,dtype=torch.long)
        target=torch.randn(batch,14)
        fa=a(obs,action,dt,identity,mask,state=sa)
        fb=b(obs,action,dt,identity,mask,state=sb)
        for field in ("mean_next","scale_next","proposed_action","movement_logit",
                      "movement_probability","observation_written",
                      "prediction_written","previous_prediction_present"):
            same(getattr(fa,field),getattr(fb,field))
        same(fa.state.world.fast,fb.state.world.fast)
        same(fa.state.world.slow,fb.state.world.slow)
        same(fa.state.last_observation,fb.state.last_observation)
        same(fa.state.last_observed_mask,fb.state.last_observed_mask)
        for field in ("ids","keys","values","occupied"):
            same(getattr(fa.state.world.memory,field),getattr(fb.state.world.memory,field))
        la=la+(fa.mean_next-target).abs().mean()+.1*fa.scale_next.mean()+.2*fa.movement_probability.square().mean()+.01*fa.proposed_action.square().mean()
        lb=lb+(fb.mean_next-target).abs().mean()+.1*fb.scale_next.mean()+.2*fb.movement_probability.square().mean()+.01*fb.proposed_action.square().mean()
        sa,sb=fa.state,fb.state
    la=la+.05*sa.world.memory.values.square().mean()
    lb=lb+.05*sb.world.memory.values.square().mean()
    la.backward();lb.backward()
    for (na,pa),(nb,pb) in zip(a.named_parameters(),b.named_parameters()):
        assert na==nb
        if pa.grad is None or pb.grad is None:
            assert pa.grad is None and pb.grad is None,na
        else:
            same(pa.grad,pb.grad)
