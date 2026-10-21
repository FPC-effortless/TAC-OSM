"""Numerical fixtures only; not synthetic capability evidence."""
import pytest
import torch
from tac_osm.me038_innovation_forecast import ARMS,InnovationMemory,build_model
from tac_osm.me036_patch_forecast import build_model as old_model


def test_parameter_contract_all_horizons():
    for h in (96,192,336,720):
        n=[sum(p.numel() for p in build_model(a,h,1).parameters()) for a in ARMS[:3]]
        assert max(n)/min(n)-1<.01

@pytest.mark.parametrize('arm',ARMS)
def test_initial_parent_forecasts_exact(arm):
    torch.set_num_threads(1);parent=old_model('ar',96,1);m=build_model(arm,96,2)
    m.initialize_ar(parent.state_dict());x=torch.randn(2,96,3)
    assert torch.equal(parent(x),m(x))


def test_innovation_removes_existing_prediction_error():
    torch.manual_seed(1);m=torch.randn(2,8,8,dtype=torch.float64)
    key=torch.nn.functional.normalize(torch.randn(2,8,dtype=torch.float64),dim=-1)
    value=torch.randn(2,8,dtype=torch.float64);beta=torch.ones(2,1,dtype=torch.float64)
    updated=InnovationMemory.innovation_write(m,key,value,beta,.99)
    assert torch.allclose(torch.bmm(updated,key[:,:,None])[:,:,0],value,atol=1e-14,rtol=1e-14)


def test_fixed_signal_memory_operator_nonexpansive():
    torch.manual_seed(3);dtype=torch.float64
    keys=InnovationMemory.address(torch.randn(50,8,dtype=dtype));beta=torch.rand(50,1,dtype=dtype)
    operator=.999*(torch.eye(8,dtype=dtype)[None]-beta[:,:,None]*keys[:,:,None]*keys[:,None,:])
    assert (torch.linalg.svdvals(operator).max(-1).values<=.999+1e-14).all()

@pytest.mark.parametrize('arm',ARMS[:3])
def test_complete_forecaster_all_paths_receive_gradients(arm):
    torch.set_num_threads(1);m=build_model(arm,96,4);opt=torch.optim.AdamW(m.parameters(),lr=.001)
    x=torch.randn(2,96,3);y=torch.randn(2,96,3)
    for _ in range(3):
        opt.zero_grad();(m(x)-y).square().mean().backward();opt.step()
    for name,p in m.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(),name
        assert p.grad.abs().sum()>0,name


def test_associative_recall_changes_native_state_updates():
    torch.set_num_threads(1);m=InnovationMemory();u=torch.randn(2,16);f=torch.zeros(2,32);s=f.clone()
    a=m.step(u,f,s,torch.zeros(2,8,8));b=m.step(u,f,s,torch.eye(8).repeat(2,1,1))
    assert not torch.equal(a[0],b[0]);assert not torch.equal(a[1],b[1])
    assert (a[0].abs()<=1).all() and (a[1].abs()<=1).all()

@pytest.mark.parametrize('arm',ARMS[:3])
def test_channel_permutation_equivariance(arm):
    torch.set_num_threads(1);m=build_model(arm,96,3)
    with torch.no_grad():m.out.weight.normal_(std=.1)
    x=torch.randn(2,96,4);perm=torch.tensor([2,0,3,1])
    assert torch.allclose(m(x[:,:,perm]),m(x)[:,:,perm],atol=2e-6,rtol=2e-6)


def test_future_patches_cannot_change_prefix_states():
    torch.set_num_threads(1);m=InnovationMemory();u=torch.randn(2,12,16);changed=u.clone();changed[:,6:]+=10
    def prefix(inputs):
        f=inputs.new_zeros(2,32);s=f.clone();a=inputs.new_zeros(2,8,8)
        for t in range(6):f,s,a=m.step(inputs[:,t],f,s,a)
        return f,s,a
    for a,b in zip(prefix(u),prefix(changed)):assert torch.equal(a,b)
