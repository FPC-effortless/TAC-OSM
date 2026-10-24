import torch
from tac_osm.variants.median_anchored_world import MedianAnchoredWorld
from tac_osm.variants.median_anchored_execution import execution_quantiles,PhysicalMedianAnchoredGraph
from tac_osm.variants.median_anchored_distribution import mixture_quantiles


def test_tail_only_execution_retains_original_quantiles_physical_units_and_state():
    torch.set_num_threads(1);torch.manual_seed(9495);m=MedianAnchoredWorld().double().eval()
    with torch.no_grad():
        m.state_mean.copy_(torch.arange(14,dtype=torch.float64));m.state_scale.fill_(2.)
        m.action_mean.fill_(.5);m.action_scale.fill_(3.)
    x=torch.rand(2,3,14);a=torch.rand_like(x);dt=torch.full((2,3,1),1/15);pixels=torch.randint(0,256,(2,3,1,32,32),dtype=torch.uint8)
    xn=((x-m.state_mean.float())/m.state_scale.float()).double();an=((a-m.action_mean.float())/m.action_scale.float()).double();initial=m.initial_state(xn[:,0])
    raw,state=m.visual_tensor_graph(xn,an,dt.double(),(pixels.float()/255).double(),initial,last_only=True)
    for u,v in zip(execution_quantiles(raw),mixture_quantiles(raw)):torch.testing.assert_close(u,v,rtol=1e-12,atol=1e-12)
    expected=tuple(q*m.state_scale+m.state_mean for q in mixture_quantiles(raw))+(raw['motion_logit'].sigmoid(),*tuple(state))
    actual=PhysicalMedianAnchoredGraph(m)(x,a,dt,pixels,tuple(initial))
    for u,v in zip(expected,actual):torch.testing.assert_close(u,v,rtol=1e-12,atol=1e-12)
