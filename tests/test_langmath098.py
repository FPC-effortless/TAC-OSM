"""Constructed no-fit source/reference checks. Never loads historical weights."""
from pathlib import Path
import importlib.util
import sys
import numpy as np
import pytest
import torch

ROOT=Path(__file__).resolve().parents[1]
HERE=ROOT/'research/mathematical-engineering/langmath098'
def load(name):
    spec=importlib.util.spec_from_file_location('langmath098_'+name,HERE/(name+'.py'))
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    return module
M=load('model');N=load('reference');O=load('objective')

def model(dtype=torch.float64):
    with torch.random.fork_rng():
        torch.manual_seed(98001)
        return M.ContextualAssociativeTAC098().to(dtype=dtype)

def params(m):return {k:v.detach().numpy().copy() for k,v in m.state_dict().items()}

@pytest.mark.parametrize('length',[1,2,3,7,16,17,65,257])
def test_affine_scan_arbitrary_state_and_zero_decay(length):
    torch.manual_seed(98);a=torch.rand(2,length,2,1,dtype=torch.float64)
    a[:,::3]=0;b=torch.randn(2,length,2,19,dtype=torch.float64);s=torch.randn(2,2,19,dtype=torch.float64)
    x=M.affine_scan(a,b,s);y=M.affine_scan(a,b,s,parallel=False)
    torch.testing.assert_close(x,y,atol=1e-12,rtol=1e-12)

@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
@pytest.mark.parametrize('length',[1,7,32,67])
def test_full_source_sequential_parallel_and_independent_numpy(dtype,length):
    m=model(dtype);tokens=torch.arange(2*length).reshape(2,length)%256;owners=('a','b')
    mask=torch.ones_like(tokens,dtype=torch.bool);mask[:,::5]=False
    with torch.no_grad():
        y,s=m(tokens,owners=owners,write_mask=mask)
        yseq,sseq=m(tokens,owners=owners,write_mask=mask,parallel=False)
    yn,sn=N.forward(params(m),tokens.numpy(),write_mask=mask.numpy())
    tol=2e-5 if dtype==torch.float32 else 2e-11
    torch.testing.assert_close(y,yseq,atol=tol,rtol=tol)
    np.testing.assert_allclose(y.numpy(),yn,atol=tol,rtol=tol)
    for name in s.tensors:
        torch.testing.assert_close(s.tensors[name],sseq.tensors[name],atol=tol,rtol=tol)
        np.testing.assert_allclose(s.tensors[name].numpy(),sn[name],atol=tol,rtol=tol)

def test_counts_state_geometry_and_all_parameters_active():
    m=model();assert m.counts()['trainable']==m.counts()['deployed']==100370
    assert m.start(('a',)).tensor_bytes()==432*8
    assert model(torch.float32).start(('a',)).tensor_bytes()==1728
    logits,_=m(torch.arange(24)[None]);(logits.square().mean()).backward()
    for name,p in m.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.norm()>0,name

@pytest.mark.parametrize('cut',[1,3,9,16])
def test_complete_chunk_state_clock_and_gradients(cut):
    m=model();x=torch.arange(23)[None];whole,sw=m(x,owners=('a',))
    first,s=m(x[:,:cut],owners=('a',));second,s=m(x[:,cut:],state=s,owners=('a',))
    torch.testing.assert_close(whole,torch.cat((first,second),1),atol=2e-11,rtol=2e-11)
    assert s.clock==sw.clock==23
    for name in s.tensors:torch.testing.assert_close(s.tensors[name],sw.tensors[name],atol=2e-11,rtol=2e-11)
    whole.square().mean().backward();full={n:p.grad.clone() for n,p in m.named_parameters()};m.zero_grad()
    torch.cat((first,second),1).square().mean().backward()
    for n,p in m.named_parameters():torch.testing.assert_close(p.grad,full[n],atol=1e-10,rtol=1e-8)

def test_source_causality_and_entity_isolation():
    m=model();x=torch.arange(38).reshape(2,19);changed=x.clone();changed[:,10:]=255;changed[1,:10]=200
    y,s=m(x,owners=('a','b'));other,_=m(changed,owners=('a','b'))
    torch.testing.assert_close(y[0,:10],other[0,:10],atol=0,rtol=0)
    single,ss=m(x[:1],owners=('a',))
    torch.testing.assert_close(single,y[:1],atol=1e-12,rtol=1e-12)
    for n in s.tensors:torch.testing.assert_close(ss.tensors[n],s.tensors[n][:1],atol=1e-12,rtol=1e-12)

def test_cone_read_bounded_and_no_write_identity_long_constructed_stream():
    m=model();x=torch.arange(1025)[None]%256
    with torch.no_grad():_,s,trace=m(x,owners=('a',),return_trace=True)
    assert trace['mass'].min()>=0 and trace['mass'].sum(-1).max()<=1+1e-12
    assert (trace['memory'].abs()<=trace['mass'][...,None,:]+1e-12).all()
    assert trace['reads'].abs().max()<=1+1e-12
    assert trace['context'].abs().max()<=1+1e-12
    before={n:v.clone() for n,v in s.tensors.items()}
    _,after=m(x[:,:7],state=s,owners=('a',),write_mask=torch.zeros(1,7,dtype=torch.bool))
    for n in ('memory','mass'):torch.testing.assert_close(after.tensors[n],before[n],atol=0,rtol=0)
    for n in s.tensors:torch.testing.assert_close(s.tensors[n],before[n],atol=0,rtol=0)

def test_contextual_keys_values_and_content_aware_comparison_are_connected():
    m=model();x=torch.tensor([[7,8,9,10]])
    _,s=m(x,owners=('a',));leaves={n:v.detach().clone().requires_grad_(True) for n,v in s.tensors.items()}
    altered=M.State098(s.owners,s.clock,leaves)
    logits,_,t=m(x[:,:1],state=altered,owners=('a',),return_trace=True)
    key_grad=torch.autograd.grad(t['keys'][0,0,0,0],leaves['context'],retain_graph=True)[0]
    value_grad=torch.autograd.grad(t['values'][0,0,0,0],leaves['context'],retain_graph=True)[0]
    compare_grad=torch.autograd.grad(t['compare_gate'].sum(),leaves['memory'],retain_graph=True)[0]
    for g in (key_grad,value_grad,compare_grad):assert torch.isfinite(g).all() and g.norm()>0
    g=torch.autograd.grad(logits.square().mean(),leaves['memory'])[0]
    eps=1e-6;i=(0,0,0,0);values=[]
    for sign in (-1,1):
        patch={n:v.detach().clone() for n,v in leaves.items()};patch['memory'][i]+=sign*eps
        out,_=m(x[:,:1],state=M.State098(s.owners,s.clock,patch),owners=('a',));values.append(out.square().mean().item())
    assert np.isclose(g[i].item(),(values[1]-values[0])/(2*eps),atol=1e-7,rtol=1e-4)

def test_parallel_and_serial_all_parameter_gradients():
    m=model();x=torch.arange(29)[None]
    y,_=m(x);y.square().mean().backward();a={n:p.grad.clone() for n,p in m.named_parameters()};m.zero_grad()
    y,_=m(x,parallel=False);y.square().mean().backward()
    for n,p in m.named_parameters():torch.testing.assert_close(p.grad,a[n],atol=1e-10,rtol=1e-8)

def test_coefficient_control_reference_and_future_gradient_unchanged():
    torch.manual_seed(98);z=torch.randn(4,9,256,dtype=torch.float64,requires_grad=True);y=torch.arange(36).reshape(4,9)
    mask=torch.zeros(4,9,dtype=torch.bool);mask[:2,-2:]=True;mask[2:,-5:]=True
    valid=torch.ones_like(mask);domains=('code','code','arc','arc')
    gradients=[]
    for weight in (0.,.3,1.,2.):
        loss=O.controlled_loss(z,y,mask,valid,domains,prefix_weight=weight)
        expected=N.controlled_loss(z.detach().numpy(),y.numpy(),mask.numpy(),valid.numpy(),domains,prefix_weight=weight)
        assert np.isclose(loss.item(),expected,atol=1e-12)
        gradients.append(torch.autograd.grad(loss,z)[0])
    for g in gradients[1:]:torch.testing.assert_close(g[mask],gradients[0][mask],atol=0,rtol=0)
    assert gradients[0][~mask].abs().sum()==0

@pytest.mark.parametrize('failure',['owner','clock','dtype','shape','nan','token','mask'])
def test_state_and_information_boundary_rejections(failure):
    m=model();x=torch.arange(7)[None];s=m.start(('a',));owners=('a',);mask=None
    if failure=='owner':owners=('b',)
    if failure=='clock':s=M.State098(s.owners,-1,s.tensors)
    if failure=='dtype':s=M.State098(s.owners,0,{**s.tensors,'mass':s.tensors['mass'].float()})
    if failure=='shape':s=M.State098(s.owners,0,{**s.tensors,'mass':s.tensors['mass'][:,:,:7]})
    if failure=='nan':s.tensors['context'][0,0]=float('nan')
    if failure=='token':x[0,0]=256
    if failure=='mask':mask=torch.ones_like(x)
    with pytest.raises(ValueError):m(x,state=s,owners=owners,write_mask=mask)


def test_source_registered_probe_adapter_boundary_and_state_derivatives():
    from tac_osm.research_probe import probe_memory,probe_dependencies,probe_entity_isolation
    m=model(torch.float32);a=load('probe_adapter').adapter098()
    sample={'tokens':torch.tensor([[1,2,3,4],[5,6,7,8]]),'owners':('a','b')}
    a.validate(m,sample)
    assert probe_memory(m,sample,a)['query_does_not_write_state']
    assert probe_dependencies(m,sample,a)['status']=='MEASURED'
    assert probe_entity_isolation(m,sample,a,tolerance=3e-5)['status']=='PASS'
    with pytest.raises(ValueError):a.validate(m,{**sample,'targets':torch.ones(2,4)})


def test_future_only_training_graph_reaches_acquisition_and_detach_removes_path():
    m=model();x=torch.arange(16).reshape(2,8);target=(x+1)%256
    future=torch.zeros_like(x,dtype=torch.bool);future[:,-2:]=True;valid=torch.ones_like(future)
    norms=[]
    for detached in (False,True):
        m.zero_grad();first,state=m(x[:,:4],owners=('a','b'))
        if detached:state=M.State098(state.owners,state.clock,{n:v.detach() for n,v in state.tensors.items()})
        later,_=m(x[:,4:],state=state,owners=('a','b'))
        loss=O.controlled_loss(torch.cat((first,later),1),target,future,valid,('code','arc'),prefix_weight=0.)
        loss.backward();norms.append(float(m.embedding.weight.grad[0].norm()))
    assert norms[0]>1e-10 and norms[1]==0
