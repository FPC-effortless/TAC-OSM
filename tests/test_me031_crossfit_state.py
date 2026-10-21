"""Numerical and isolation invariants, not empirical state-value evidence."""
import pytest
import torch
from tac_osm.me031_crossfit_state import CrossfitNativeEncoder, ResidualProbe, role_indices
from tac_osm.me028_nonlinear_readout import NativeNonlinearObservableReadout


def tensors():
    torch.set_num_threads(1);torch.manual_seed(31031)
    return (torch.randint(16000,(2,128)),torch.rand(2,128,3),
            torch.randint(16000,(2,)),torch.rand(2,1))


def test_user_roles_are_disjoint_complete_and_rotated():
    users=torch.tensor([0,0,16,32,32,48,64,80])
    targets=[]
    for r in range(3):
        roles=role_indices(users,r)
        assert sorted(torch.cat(roles).tolist())==list(range(len(users)))
        sets=[set(users[ix].tolist()) for ix in roles]
        assert all(not sets[i]&sets[j] for i,j in ((0,1),(0,2),(1,2)))
        targets.extend(roles[2].tolist())
    assert sorted(targets)==list(range(len(users)))
    with pytest.raises(ValueError):role_indices(torch.tensor([1]),0)


def test_exported_features_preserve_native_and_prefix_causality():
    args=tensors();pop=torch.full((16000,),.4)
    torch.manual_seed(3);old=NativeNonlinearObservableReadout(.5,.2,pop)
    torch.manual_seed(3);new=CrossfitNativeEncoder(.5,.2,pop)
    assert sum(p.numel() for p in new.parameters())==135377
    assert all(torch.equal(v,new.state_dict()[k]) for k,v in old.state_dict().items())
    f=new.prefix_features(*args)
    assert f.shape==(2,4,91)
    assert torch.allclose(f[:,-1,:64],new.latent_features(*args[:2]))
    assert torch.allclose(f,torch.jit.script(new).prefix_features(*args),atol=1e-6)
    logits=new.predict_features(f[...,:64],f[...,64:90],torch.sigmoid(f[...,90:91])).squeeze(-1)
    assert torch.allclose(logits,new.prefix_logits(*args),atol=1e-6)
    for depth,col in ((32,0),(64,1),(96,2)):
        i,m,a,d=[t.clone() for t in args]
        i[:,depth+1:]=torch.randint(16000,i[:,depth+1:].shape)
        m[:,depth+1:]=torch.rand_like(m[:,depth+1:]);m[:,depth,1]=1-m[:,depth,1]
        assert torch.allclose(new.prefix_features(i,m,a,d)[:,col],f[:,col],atol=1e-6)


@pytest.mark.parametrize('state,count',[(False,1205),(True,1197)])
def test_probe_counts_initial_offset_active_gradients_and_access(state,count):
    torch.manual_seed(31)
    net=ResidualProbe(state)
    assert sum(p.numel() for p in net.parameters())==count
    x=torch.randn(3,4,91,requires_grad=True);offset=torch.randn(3,4)
    assert torch.equal(net(x,offset),offset)
    with torch.no_grad():net.output.weight.normal_()
    out=net(x,offset)
    assert torch.allclose(out,torch.jit.script(net)(x,offset))
    out.square().sum().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.norm()>0 for p in net.parameters())
    if state:assert x.grad[...,:64].norm()>0
    else:assert x.grad[...,:64].abs().sum()==0
    assert x.grad[...,90].abs().sum()==0  # only offset carries original prior


def test_source_subset_preserves_scalar_provenance():
    from scripts.me031_crossfit_research import subset
    source={'items':torch.arange(6).reshape(3,2),'users':[0,16,32],
            'eligible_target_count':100,'sampling_stride':2}
    result=subset(source,torch.tensor([0,2]))
    assert result['users']==[0,32]
    assert torch.equal(result['items'],source['items'][[0,2]])
    assert result['eligible_target_count']==100 and result['sampling_stride']==2
