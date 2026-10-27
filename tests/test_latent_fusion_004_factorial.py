"""LATFUSE004 constructed operator tests. Not intelligence/real prediction results."""
import pytest
import torch
import torch.nn.functional as F

from tac_osm.latent_fusion_004_factorial import ARMS,FactorialFusion,contract,partitioned_update,head_weights
from tac_osm.latent_fusion_003 import FusedLatentFourHead


@pytest.mark.parametrize("arm",list(ARMS))
def test_full_256_matrix_cells_fp32_state_and_100k_parity(arm):
    model=FactorialFusion(arm)
    c=contract(arm)
    assert c["matrix_cells"]==256 and c["per_stream_fp32_scalars"]==1696
    assert c["per_stream_nominal_bytes"]==6800
    assert 95000<=c["active_parameters"]<=105000
    assert c["active_parameters"] in (99747,99261)
    assert (99747-99261)/99747 < .01
    z=model.initial(3)
    assert z.associative.shape==(3,16,16)
    assert z.entities.shape==(3,16,80)
    assert all(v.dtype==torch.float32 for v in z[:4])
    assert z.active.dtype==torch.bool


@pytest.mark.parametrize("arm",list(ARMS))
def test_source_no_observation_does_not_update_associative_or_source_slots(arm):
    torch.manual_seed(402)
    model=FactorialFusion(arm)
    s=model.initial(2)
    x=torch.randn(2,13)
    mask=torch.zeros(2,13,dtype=torch.bool)
    out=model.step(s,x,mask,torch.full_like(x,10),torch.ones_like(x),torch.tensor([0,1]))
    assert torch.equal(out.associative,s.associative)
    assert torch.equal(out.entities,s.entities)
    assert torch.equal(out.active,s.active)


@pytest.mark.parametrize("arm",list(ARMS))
def test_forecast_is_readonly_and_candidate_query_permutation(arm):
    torch.manual_seed(404)
    model=FactorialFusion(arm)
    s=model.initial(2)
    for _ in range(3):
        x=torch.randn(2,13)
        mask=torch.ones_like(x,dtype=torch.bool)
        s=model.step(s,x,mask,torch.zeros_like(x),torch.zeros_like(x),torch.tensor([0,1]))
    before=[v.detach().clone() for v in s]
    q=torch.randn(2,4,6)
    m,sd=model.forecast(s,torch.tensor([0,1]),q)
    a,b=model.forecast(s,torch.tensor([0,1]),q[:,[3,1,0,2]])
    torch.testing.assert_close(m[:,[3,1,0,2]],a,atol=5e-7,rtol=1e-5)
    torch.testing.assert_close(sd[:,[3,1,0,2]],b,atol=5e-7,rtol=1e-5)
    assert all(torch.equal(v,x) for v,x in zip(s,before))


def test_four_head_hard_is_existing_LATFUSE003_step_and_readout():
    torch.manual_seed(554)
    original=FusedLatentFourHead()
    torch.manual_seed(554)
    factorial=FactorialFusion("4h-hard")
    for name,p in original.state_dict().items():
        torch.testing.assert_close(p,factorial.state_dict()[name],atol=0,rtol=0)
    for step in range(4):
        x=torch.randn(3,13)
        obs=torch.ones(3,13,dtype=torch.bool)
        ts=torch.zeros(3,13)
        idx=torch.arange(3)
        if step==0:
            a=original.initial(3);b=factorial.initial(3)
        a=original.step(a,x,obs,ts,ts,idx)
        b=factorial.step(b,x,obs,ts,ts,idx)
        for i,j in zip(a,b):
            torch.testing.assert_close(i,j,atol=0,rtol=0)
    q=torch.randn(3,1,6)
    for i,j in zip(original.forecast(a,idx,q),factorial.forecast(b,idx,q)):
        torch.testing.assert_close(i,j,atol=0,rtol=0)


@pytest.mark.parametrize("head_count,soft",[(2,False),(2,True),(4,False),(4,True)])
def test_same_matrix_capacity_hard_exact_isolation_soft_all_head_write(head_count,soft):
    torch.manual_seed(110)
    m=torch.randn(2,16,16)
    key=torch.randn(2,16)
    value=torch.randn(2,16)
    a=torch.full((2,1),.994);b=torch.full((2,1),.2)
    logit=torch.tensor([[7.,-9.,0.,-1.],[7.,-9.,0.,-1.]])[:,:head_count]
    out,w,sel=partitioned_update(
        m,key,value,a,b,logit,torch.ones(2,dtype=torch.bool),
        torch.zeros(head_count),head_count=head_count,soft=soft)
    assert out.shape==m.shape and torch.isfinite(out).all()
    torch.testing.assert_close(w.sum(-1),torch.ones(2),atol=1e-6,rtol=0)
    prior=m.reshape(2,16,head_count,16//head_count)
    after=out.reshape(2,16,head_count,16//head_count)
    if soft:
        assert sel.all()
        assert any(not torch.equal(prior[:,:,h,:],after[:,:,h,:]) for h in range(head_count))
    else:
        assert (sel.sum(-1)==head_count//2).all()
        for h in range(head_count):
            if not sel[:,h].any():
                assert torch.equal(prior[:,:,h,:],after[:,:,h,:])


@pytest.mark.parametrize("arm",list(ARMS))
def test_gradients_to_write_and_read_router_and_all_trainable_source_parameters(arm):
    torch.manual_seed(110)
    m=FactorialFusion(arm)
    state=m.initial(3)
    e=[];val=[];mask=[]
    for t in range(6):
        x=torch.randn(3,13)+t*1.2
        visible=torch.ones_like(x,dtype=torch.bool)
        u=m.encode(x,visible,torch.zeros_like(x),torch.zeros_like(x))
        e.append(u);val.append(x);mask.append(visible)
        state=m.step(state,x,visible,torch.zeros_like(x),torch.zeros_like(x),torch.arange(3))
    q=torch.randn(3,1,6)
    pred,sd=m.forecast(state,torch.arange(3),q)
    loss=pred.square().mean()+sd.square().mean()+.25*m.common_key_loss(torch.stack(e,1),torch.stack(val,1),torch.stack(mask,1))+.03*m.common_head_balance(torch.stack(e,1),torch.stack(mask,1))
    loss.backward()
    assert torch.isfinite(loss)
    for name,p in m.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(),name


def test_hard_writes_are_locally_discontinuous_soft_writes_receive_all_head_gradients():
    torch.manual_seed(323)
    for arm in ("2h-soft","4h-soft"):
        model=FactorialFusion(arm)
        state=model.initial(2)
        # isolated source-grad on actual memory final state, not auxiliary routing
        x=torch.randn(2,13)
        valid=torch.ones_like(x,dtype=torch.bool)
        out=model.step(state,x,valid,torch.zeros_like(x),torch.zeros_like(x),torch.tensor([0,1]))
        out.associative.square().sum().backward()
        g=model.write_router.weight.grad
        assert g is not None and torch.isfinite(g).all()
        assert g.abs().sum()>0


def test_key_loss_ignores_missing_source_and_unknown_targets():
    for arm in ("2h-hard","2h-soft","4h-hard","4h-soft"):
        m=FactorialFusion(arm)
        u=torch.randn(2,4,80,requires_grad=True)
        values=torch.randn(2,4,13)
        unseen=torch.zeros_like(values,dtype=torch.bool)
        loss=m.common_key_loss(u,values,unseen)
        assert float(loss)==0.
        loss.backward()
        assert u.grad is not None and torch.equal(u.grad,torch.zeros_like(u.grad))


def test_unknown_variant_and_dtype_mismatch_fail_closed():
    with pytest.raises(ValueError):FactorialFusion("unknown")
    with pytest.raises(ValueError):
        partitioned_update(torch.zeros(1,16,16,dtype=torch.float64),
             torch.zeros(1,16),torch.zeros(1,16),
             torch.ones(1,1),torch.ones(1,1),torch.zeros(1,2),
             torch.ones(1,dtype=torch.bool),torch.zeros(2),head_count=2,soft=True)
