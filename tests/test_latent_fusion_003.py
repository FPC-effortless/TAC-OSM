"""Constructed LATFUSE003 operator guards; NOT real-world predictive evidence."""
import torch
import pytest
from torch.nn import functional as F
from tac_osm.latent_fusion_002 import FusedLatentFixedState
from tac_osm.latent_fusion_003 import (
    FusedLatentFourHead,exact_state_contract,four_head_delta,_router
)


def _model():
    torch.manual_seed(31004)
    return FusedLatentFourHead()


def test_exact_state_and_living_parameter_parity():
    old=FusedLatentFixedState()
    new=_model()
    assert [x.shape for x in old.initial(2)]==[x.shape for x in new.initial(2)]
    assert [x.dtype for x in old.initial(2)]==[x.dtype for x in new.initial(2)]
    c=exact_state_contract(new)
    assert c["nominal_bytes"]==6800 and c["matrix_cells"]==256
    assert c["trainable_parameters"]==99747
    assert (99747-98775)/99747 < .01
    assert all(p.requires_grad and p.dtype==torch.float32 for p in new.parameters())


def test_multihead_never_truncates_original_matrix_or_adds_cache():
    m=_model()
    s=m.initial(4)
    assert s.associative.shape==(4,16,16)
    assert len(s)==5
    assert not any(getattr(m,name,None) is not None
        for name in ("attention_kv_cache","token_cache","history_cache"))


def test_unselected_heads_remain_bitwise_untouched():
    matrix=torch.arange(2*16*16,dtype=torch.float32).reshape(2,16,16)/100
    k=torch.ones(2,16)
    v=torch.randn(2,16)
    retention=torch.full((2,1),.7)
    beta=torch.full((2,1),.5)
    logit=torch.tensor([[11.,10.,-9.,-11.],[11.,10.,-9.,-11.]])
    result,selected,weights=four_head_delta(matrix,k,v,retention,beta,logit,
                          torch.tensor([True,True]),torch.zeros(4))
    assert selected.tolist()==[[True,True,False,False]]*2
    a=matrix.reshape(2,16,4,4)
    b=result.reshape(2,16,4,4)
    assert torch.equal(a[:,:,2:,:],b[:,:,2:,:])
    assert not torch.equal(a[:,:,:2,:],b[:,:,:2,:])
    torch.testing.assert_close(weights.sum(-1),torch.ones(2))


def test_source_missing_entirely_never_erases_matrix_or_writes_entity():
    model=_model()
    s=model.initial(2)
    x=torch.randn(2,13)
    obs=torch.zeros_like(x,dtype=torch.bool)
    previous=torch.randn(2,13)
    z=model.step(s,x,obs,torch.full_like(x,19),previous,torch.tensor([0,1]))
    assert torch.equal(z.associative,s.associative)
    assert torch.equal(z.entities,s.entities)
    assert torch.equal(z.active,s.active)


def test_transport_matches_right_affine_in_each_independent_head():
    torch.manual_seed(23)
    matrix=torch.randn(3,16,16)
    k=torch.randn(3,16)
    v=torch.randn(3,16)
    a=torch.full((3,1),.992)
    b=torch.full((3,1),.4)
    scores=torch.randn(3,4)
    out,sel,weights=four_head_delta(matrix,k,v,a,b,scores,
                                    torch.ones(3,dtype=torch.bool),torch.zeros(4))
    M=matrix.reshape(3,16,4,4).permute(0,2,1,3)
    K=F.normalize(k.reshape(3,4,4),dim=-1)
    block=[]
    for h in range(4):
        keep=a if sel[:,h].all() else torch.where(sel[:,h,None],a,torch.ones_like(a))
        beta=(b[:,0]*weights[:,h]*sel[:,h]).reshape(3,1)
        # P=keep(I-beta kk^T), Q=beta v k^T
        eye=torch.eye(4)[None].expand(3,-1,-1)
        P=keep[:,:,None]*(eye-beta[:,:,None]*K[:,h,:,None]*K[:,h,None,:])
        Q=beta[:,:,None]*v[:,:,None]*K[:,h,None,:]
        block.append(M[:,h]@P+Q)
    expected=torch.stack(block,1).permute(0,2,1,3).reshape(3,16,16)
    torch.testing.assert_close(out,expected,atol=1e-5,rtol=1e-5)


def test_four_head_forecast_query_permutation_read_only():
    model=_model()
    s=model.initial(2)
    for _ in range(4):
        x=torch.randn(2,13)
        obs=torch.ones_like(x,dtype=torch.bool)
        s=model.step(s,x,obs,torch.zeros_like(x),torch.zeros_like(x),torch.tensor([0,1]))
    before=[v.clone() for v in s]
    q=torch.randn(2,5,6)
    with torch.no_grad():
        m,sigma=model.forecast(s,torch.tensor([0,1]),q)
        m2,sigma2=model.forecast(s,torch.tensor([0,1]),q[:,[3,1,4,0,2]])
    torch.testing.assert_close(m[:,[3,1,4,0,2]],m2,atol=0,rtol=0)
    torch.testing.assert_close(sigma[:,[3,1,4,0,2]],sigma2,atol=0,rtol=0)
    assert all(torch.equal(a,b) for a,b in zip(s,before))


def test_all_new_router_parameters_receive_gradient_from_source_compatible_objective():
    model=_model()
    s=model.initial(4)
    embeddings=[]
    seen=[]
    raw=[]
    torch.manual_seed(123)
    for t in range(8):
        x=torch.randn(4,13)+t*.6
        mask=torch.ones_like(x,dtype=torch.bool)
        u=model.encode(x,mask,torch.zeros_like(x),torch.zeros_like(x))
        embeddings.append(u);seen.append(mask);raw.append(x)
        s=model.step(s,x,mask,torch.zeros_like(x),torch.zeros_like(x),
                     torch.arange(4))
    pred,sigma=model.forecast(s,torch.arange(4),torch.randn(4,1,6))
    sep=model.source_key_separation_loss(torch.stack(embeddings,1),torch.stack(raw,1),torch.stack(seen,1))
    balance=model.head_utilization_loss(torch.stack(embeddings,1),torch.stack(seen,1))
    loss=pred.square().mean()+sigma.square().mean()+.02*sep+.005*balance
    assert torch.isfinite(loss)
    loss.backward()
    for name in ("write_router","read_router","key_map","forecast_head","embed"):
        for p in getattr(model,name).parameters():
            assert p.grad is not None and bool(torch.isfinite(p.grad).all()),name
    assert model.log_head_temperature.grad is not None
    assert bool(torch.isfinite(model.log_head_temperature.grad).all())


def test_temperature_changes_only_competition_among_registered_heads():
    scores=torch.tensor([[3.,2.,1.,0.]])
    a,m=_router(scores,torch.zeros(4),top_k=2)
    b,n=_router(scores,torch.full((4,),.5),top_k=2)
    assert torch.equal(m,n)
    assert (a[0,2:]==0).all()
    assert not torch.allclose(a,b)
    assert torch.isfinite(b).all()


def test_key_loss_does_not_require_artificial_labels_or_future_targets():
    m=_model()
    x=torch.zeros(2,3,13)
    seen=torch.zeros(2,3,13,dtype=torch.bool)
    u=torch.randn(2,3,80,requires_grad=True)
    loss=m.source_key_separation_loss(u,x,seen)
    assert loss.item()==0
    loss.backward()
    assert u.grad is not None and not u.grad.abs().any()


def test_numerical_bad_state_rejected_instead_of_silent_memory_corruption():
    with pytest.raises(ValueError):
        four_head_delta(torch.zeros(1,16,16,dtype=torch.float64),
            torch.zeros(1,16,dtype=torch.float64),torch.zeros(1,16,dtype=torch.float64),
            torch.ones(1,1,dtype=torch.float64),torch.ones(1,1,dtype=torch.float64),
            torch.zeros(1,4,dtype=torch.float64),torch.ones(1,dtype=torch.bool),torch.zeros(4))
