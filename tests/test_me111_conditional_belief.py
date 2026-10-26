import torch
from tac_osm.variants.conditional_belief_sensor_world import ConditionalBeliefSensorWorld,conditional_scales
from tac_osm.variants.residual_role_sensor_world import ResidualRoleSensorWorld
from tac_osm.conditional_belief_learning import reconstruction_mean_masks
from tac_osm.typed_role_learning import uncertainty_masks
from scripts.me111_objective import nll_elements


def test_live_roles_are_disjoint_and_use_original_budget():
    m=ConditionalBeliefSensorWorld();r=reconstruction_mean_masks(m);u=uncertainty_masks(m)
    assert sum(p.numel() for p in m.parameters())==104663
    assert sum(int(x.sum()) for x in r)==696 and sum(int(x.sum()) for x in u)==4410
    assert not any(bool((a&b).any()) for a,b in zip(r,u))


def test_source_scale_formula_gradient_below_old_floor():
    v=torch.ones(2,6,dtype=torch.float64,requires_grad=True)
    raw=torch.full((2,12),-1.,dtype=torch.float64,requires_grad=True)
    w=torch.exp(torch.log(torch.tensor(4.))*raw.tanh())
    l,r,b=conditional_scales(v,v,torch.ones(6),torch.ones(2,6),w)
    assert bool((l>0).all()) and bool((l<.8*b).all())
    assert bool((torch.autograd.grad(l.sum()+r.sum(),raw)[0]>0).all())


def test_probability_elements_equal_original_censored_nll():
    m=ConditionalBeliefSensorWorld().double()
    from tac_osm.variants.median_split_sensor_world import MedianSplitForecast
    f=MedianSplitForecast(torch.zeros(2,6,dtype=torch.float64),torch.ones(2,6),torch.full((2,6),2.))
    truth=torch.arange(12,dtype=torch.float64).reshape(2,6);mask=torch.ones(2,6,dtype=torch.bool)
    torch.testing.assert_close(nll_elements(m,f,truth).mean(),m.nll(f,truth,mask))
