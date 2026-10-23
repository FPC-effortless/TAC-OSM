"""Engineering only: prospective risk-shrinkage invariants, not learned skill."""
import math,json
from pathlib import Path
import pytest,torch
from tac_osm.variants.risk_shrinkage_world import ShrinkageWorld
from tac_osm.variants.kinematic_world import KinematicWorld,dense_loss
from scripts.me079_train_world import integrity


def test_actual_sizes_and_prefit_manifest():
    reg=integrity();assert len(reg['arms'])==15
    counts=[]
    for arm in ('native','gru','transformer'):
        m=ShrinkageWorld(arm);n=sum(p.numel() for p in m.parameters());counts.append(n)
        assert 95000<=n<=105000
        assert n==sum(p.numel() for p in KinematicWorld(arm).parameters())+42
        assert all(r['trainable_parameters']==n for r in reg['arms'] if r['architecture']==arm)
    assert max(counts)/min(counts)-1<=.01


@pytest.mark.parametrize('arm',['native','gru','transformer'])
def test_shrinkage_cannot_increase_raw_correction_magnitude_and_preserves_order(arm):
    torch.manual_seed(9);m=ShrinkageWorld(arm)
    raw=KinematicWorld(arm);raw.load_state_dict({k:v for k,v in m.state_dict().items() if k!='shrinkage_scale'})
    x=torch.randn(2,8,14);a=torch.randn_like(x)
    before=raw(x,a);after=m(x,a)
    assert ((after['median']-x[:,:,None]).abs()<=(before['median']-x[:,:,None]).abs()+1e-6).all()
    assert (after['lower']<=after['median']).all() and (after['median']<=after['upper']).all()
    torch.testing.assert_close(after['upper']-after['lower'],before['upper']-before['lower'],atol=1e-6,rtol=1e-5)
    loss=dense_loss(after,torch.randn(2,8,3,14),x,torch.ones(2,8,3,dtype=torch.bool),torch.ones(2),m.state_scale,.07)
    loss.backward()
    assert m.shrinkage_scale.grad is not None and torch.isfinite(m.shrinkage_scale.grad).all()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())
