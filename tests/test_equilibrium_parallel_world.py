import pytest
import torch
from torch import nn
from tac_osm.variants.equilibrium_parallel_world import EquilibriumParallelWorld
from tac_osm.variants.horizon_query_world import horizon_distribution_loss
from tac_osm.variants.reusable_execution import mixture_quantiles


def test_complete_budget_gradients_and_causality():
    torch.manual_seed(82);m=EquilibriumParallelWorld().double()
    x=torch.randn(2,32,14,dtype=torch.double)*.01;a=torch.randn_like(x)
    p=m(x,a);assert sum(v.numel() for v in m.parameters())==104539
    y=x[:,:,None]+torch.randn(2,32,3,14,dtype=torch.double)*.01
    loss=horizon_distribution_loss(p,y,x,torch.ones(2,32,3,dtype=torch.bool),torch.ones(2),m.state_scale,.01)
    loss.backward()
    for name,v in m.named_parameters():
        assert v.grad is not None and torch.isfinite(v.grad).all() and v.grad.abs().sum()>0,name
    altered=x.clone();altered[:,17:]+=20
    changed=m(altered,a)
    for k in p:torch.testing.assert_close(p[k][:,:17],changed[k][:,:17])


def test_equilibrium_is_preserved_for_arbitrary_learned_head_values():
    m=EquilibriumParallelWorld().double()
    with torch.no_grad():m.heads.bias.uniform_(-5,5)
    x=torch.ones(2,32,14,dtype=torch.double);a=torch.zeros_like(x)
    p=m(x,a)
    torch.testing.assert_close(p['current'],x[:,:,None].expand(2,32,3,14),rtol=0,atol=0)
    assert p['moving_delta'].abs().max()==0
    quantiles=mixture_quantiles({k:v[:,-1] for k,v in p.items()})
    torch.testing.assert_close(quantiles[1],x[:,-1,None].expand(2,3,14),rtol=0,atol=1e-9)


def test_parallel_filters_clocks_and_stream_state_match_full_prefix():
    torch.manual_seed(84);m=EquilibriumParallelWorld().double();x=torch.randn(2,32,14,dtype=torch.double)*.01;a=torch.randn_like(x)
    x[:,5:9]=x[:,4:5]
    full,end=m(x,a,return_state=True);left,state=m(x[:,:13],a[:,:13],return_state=True)
    right,state=m(x[:,13:],a[:,13:],state,True)
    for k in full:torch.testing.assert_close(full[k],torch.cat((left[k],right[k]),1),rtol=1e-9,atol=1e-10)
    stepped=[];state=None
    for t in range(32):
        p,state=m.step(x[:,t],a[:,t],state);stepped.append(p)
    for k in full:torch.testing.assert_close(full[k],torch.stack([p[k] for p in stepped],1),rtol=1e-9,atol=1e-10)
    for u,v in zip(state,end):torch.testing.assert_close(u,v,rtol=1e-9,atol=1e-10)


def test_tensor_graph_exports_without_data_dependent_python():
    class Graph(nn.Module):
        def __init__(self):super().__init__();self.model=EquilibriumParallelWorld()
        def forward(self,x,a):return self.model.tensor_graph(x,a)[0]
    graph=Graph().eval();x=torch.randn(2,8,14)*.01;a=torch.randn_like(x)
    exported=torch.export.export(graph,(x,a),strict=True)
    expected=graph(x,a);actual=exported.module()(x,a)
    for k in expected:torch.testing.assert_close(expected[k],actual[k])
