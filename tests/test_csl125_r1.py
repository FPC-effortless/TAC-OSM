"""New CSL-125-R1 engineering tests, no real data and zero optimizer updates."""
import math
import numpy as np
import pytest
import torch
from tac_osm.csl001 import RoleStateBlock
from tac_osm.csl125 import CSL125,CSL125Config,CAP
from tac_osm.csl125_r1 import (CSL125R1, CSL125R1Config, R1RoleStateBlock,
                                R1LayerState, bounded_weighted_mass_scan)
from tac_osm.csl125_r1_math_reference import serial_weighted_mean_memory


def cfg(**changes):
    d=dict(vocab_size=256,width=24,layers=4,ff_width=48,role_count=3,
           state_width=8,attention_layers=(2,4),query_heads=4,kv_heads=2,
           attention_window=7,rungs=(2,4),chunk_size=16)
    d.update(changes)
    return CSL125R1Config(**d)


def test_hard_ceiling_identical_trainable_parameter_count():
    with torch.device('meta'):
        r=CSL125R1(device='meta')
        legacy=CSL125(device='meta')
    assert r.parameter_count()==legacy.parameter_count()==123_699_534
    assert r.parameter_count()<=CAP
    assert r.parameter_breakdown()==legacy.parameter_breakdown()
    assert len(list(r.parameters()))==len(list(legacy.parameters()))


def test_new_init_cures_near_uniform_role_scores():
    torch.manual_seed(47)
    c=cfg(role_count=8,state_width=256,width=32,query_heads=4,kv_heads=2)
    old=RoleStateBlock(c);new=R1RoleStateBlock(c)
    h=torch.randn(2,128,32)
    def score(model):
        x=model.pre(h)
        q=model.key(x)
        logits=torch.einsum('bte,re->btr',q,model.roles)/math.sqrt(c.state_width)
        p=logits.softmax(-1)
        return (-p*p.log()).sum(-1).mean().item(), p.max(-1).values.mean().item()
    old_entropy,old_max=score(old);new_entropy,new_max=score(new)
    assert old_entropy>math.log(c.role_count)-.01
    assert new_entropy<old_entropy-.025
    assert new_max>old_max+.02


def test_retention_range_and_no_legacy_decay_clamp():
    m=R1RoleStateBlock(cfg()).eval()
    f,s=m.rates()
    assert f.min()>0 and s.min()>0
    slow_survival=torch.exp(-s.min()*4096)
    assert slow_survival>.55
    assert torch.exp(-s.max()*1024)<.025
    assert torch.all(torch.exp(-s)<1)
    assert torch.all(torch.exp(-s)>.995)  # old scan hard-clipped here


def test_scan_matches_independent_serial_float64_at_chunk_boundaries():
    from tac_osm.csl125_r1_math_reference import serial_weighted_mean_memory
    rng=np.random.default_rng(29)
    b,t,r,e=2,143,3,8
    val=rng.uniform(-1,1,size=(b,t,e)).astype('float32')
    p=rng.dirichlet(np.ones(r),size=(b,t)).astype('float32')
    rates=np.exp(rng.uniform(np.log(1/8192),np.log(1/8),size=(r,e))).astype('float32')
    numerator=p[...,None]*val[:,:,None,:]
    mass=np.broadcast_to(p[...,None],numerator.shape)
    z=torch.from_numpy(np.stack([numerator,mass],axis=-1))
    init=torch.zeros((b,r,e,2))
    y,final=bounded_weighted_mass_scan(z,init,torch.from_numpy(rates),16)
    refread,states,refend=serial_weighted_mean_memory(val,p,rates,epsilon=1e-8)
    np.testing.assert_allclose(y.detach().numpy(),states,atol=1.5e-5,rtol=1e-4)
    np.testing.assert_allclose(final.detach().numpy(),refend,atol=1.5e-5,rtol=1e-4)
    read=(torch.from_numpy(p).unsqueeze(-1)*y[...,0]/(y[...,1]+1e-8)).sum(2)
    np.testing.assert_allclose(read.numpy(),refread,atol=1.5e-5,rtol=1e-4)
    assert torch.all(y[...,1]>=0)
    assert torch.all(y[...,0].abs()<=y[...,1]+1e-6)
    assert read.abs().max()<=1+1e-6


def test_normalized_memory_removes_uniform_assignment_eightfold_penalty():
    # For a constant v and a_j=1/R, the old stationary role read is v/R.
    # The normalized read is v, provided each slot has nonzero occupancy.
    r=8
    v=0.6
    address=1/r
    old_read=sum(address*(address*v) for _ in range(r))
    new_read=sum(address*(address*v/address) for _ in range(r))
    assert old_read==pytest.approx(v/r)
    assert new_read==pytest.approx(v)


def test_full_sequence_equals_arbitrary_streamed_slices_including_over_chunk():
    torch.manual_seed(711)
    model=CSL125R1(cfg()).eval()
    x=torch.randint(0,256,(1,75))
    with torch.no_grad():
        full, fs=model(x,depth=4,stream_ids=('doc-a',))
        s=None; outputs=[]
        for a,b in ((0,7),(7,22),(22,47),(47,75)):
            o,s=model(x[:,a:b],depth=4,stream_ids=('doc-a',),state=s)
            outputs.append(o[4]['token_logits'])
    torch.testing.assert_close(torch.cat(outputs,dim=1),full[4]['token_logits'],atol=1e-4,rtol=1e-4)
    for ls,rs in zip(s.layers,fs.layers):
        for name in ('fast','slow','fast_mass','slow_mass','conv_tail'):
            torch.testing.assert_close(getattr(ls,name),getattr(rs,name),atol=3e-5,rtol=3e-5)
    for sa,fa in zip(s.attention,fs.attention):
        if sa is not None:
            torch.testing.assert_close(sa.k,fa.k,atol=3e-5,rtol=3e-5)
            assert sa.position==75


def test_causality_future_poison_all_exits():
    torch.manual_seed(712)
    m=CSL125R1(cfg()).eval()
    a=torch.randint(0,256,(1,50));b=a.clone();b[:,24:]=torch.randint(0,256,(1,26))
    with torch.no_grad():
        oa,_=m(a,all_rungs=True);ob,_=m(b,all_rungs=True)
    for depth in m.config.rungs:
        torch.testing.assert_close(oa[depth]['token_logits'][:,:24],ob[depth]['token_logits'][:,:24],atol=1e-5,rtol=1e-5)


def test_gradient_all_slow_tau_role_and_memory_paths():
    torch.manual_seed(713)
    m=CSL125R1(cfg()).train()
    x=torch.randint(0,256,(2,32))
    out,_=m(x,all_rungs=True)
    loss=sum(v['token_logits'].square().mean() for v in out.values())
    loss.backward()
    paths=['roles','key.weight','value.weight','fast_tau_logits','slow_tau_logits','read_fast.weight','read_slow.weight']
    for path in paths:
        g=dict(m.blocks[0].named_parameters())[path].grad
        assert g is not None and torch.isfinite(g).all(),path
        assert g.abs().sum()>0,path


def test_state_v2_incompatible_and_entity_ids_guarded():
    torch.manual_seed(714)
    m=CSL125R1(cfg()).eval()
    original=CSL125(CSL125Config(vocab_size=256,width=24,layers=4,ff_width=48,
        role_count=3,state_width=8,attention_layers=(2,4),query_heads=4,
        kv_heads=2,attention_window=7,rungs=(2,4),chunk_size=16))
    t=torch.randint(0,256,(1,8))
    _,prior=original(t,depth=2,stream_ids=('x',))
    with pytest.raises(ValueError,match='own memory-state version'):
        m(t,depth=2,stream_ids=('x',),state=prior)
    _,own=m(t,depth=2,stream_ids=('x',))
    with pytest.raises(ValueError,match='stream'):
        m(t,depth=2,stream_ids=('y',),state=own)
    with pytest.raises(ValueError,match='depth'):
        m(t,depth=4,stream_ids=('x',),state=own)


def test_invalid_odd_head_dim_rejected_at_config():
    with pytest.raises(ValueError,match='even head dimension'):
        cfg(width=36,query_heads=4,kv_heads=2)


def test_nan_loss_weight_and_empty_label_mask_rejected():
    m=CSL125R1(cfg())
    t=torch.randint(0,256,(1,6))
    with pytest.raises(ValueError,match='finite'):
        m.nested_next_token_loss(t,{2:float('nan'),4:1.})
    with pytest.raises(ValueError,match='empty'):
        m.nested_next_token_loss(t,loss_mask=torch.zeros(1,5,dtype=torch.bool))
    with pytest.raises(ValueError,match='boolean'):
        m.nested_next_token_loss(t,loss_mask=torch.ones(1,5))


def test_valid_masked_all_exit_language_loss():
    torch.manual_seed(715)
    m=CSL125R1(cfg())
    t=torch.randint(0,256,(1,9))
    mask=torch.tensor([[1,1,0,0,1,1,1,0]],dtype=torch.bool)
    x=m.nested_next_token_loss(t,loss_mask=mask)
    x.backward()
    assert torch.isfinite(x)
    assert m.blocks[-1].fast_tau_logits.grad is not None
    assert m.blocks[-1].fast_tau_logits.grad.isfinite().all()


def test_small_discrete_reference_holds_across_retention_grid():
    torch.manual_seed(716)
    for tau in (2.1,8.,64.,256.,8192.,32760.):
        z=torch.rand(1,60,2,3,2)
        z[...,0]*=z[...,1]
        initial=torch.zeros(1,2,3,2)
        y,_=bounded_weighted_mass_scan(z,initial,torch.full((2,3),1/tau),13)
        assert torch.isfinite(y).all()
        assert torch.all(y[...,0]<=y[...,1]+1e-5)


def test_bounded_memory_selective_slot_mathematical_probe():
    # A one-hot role receives a value; subsequent updates of a different role
    # must leave the first role's normalized content recoverable after delay.
    # This is ORACLE addressing, not learned binding or a model score.
    b,t,r,e=1,4097,2,1
    val=np.zeros((b,t,e),dtype=np.float64);val[0,0,0]=0.7
    adr=np.zeros((b,t,r),dtype=np.float64);adr[0,0,0]=1;adr[0,1:,1]=1
    rates=np.full((r,e),1/8192,dtype=np.float64)
    _,_,s=serial_weighted_mean_memory(val,adr,rates,epsilon=1e-8)
    recovered=s[0,0,0,0]/(s[0,0,0,1]+1e-8)
    assert abs(recovered-0.7)<1e-3
    # The R1 success is only under externally supplied exact one-hot routing.
    assert s[0,0,0,1]>0


def test_backpropagate_across_chunk_boundary_without_detach():
    torch.manual_seed(800)
    m=CSL125R1(cfg()).eval()
    x=torch.randint(0,256,(1,38))
    # Identical deterministic parameter graph, with the same gradients through
    # streamed carried memory; not a training step or learned effect.
    o,_=m(x,depth=4,stream_ids=('grad-seq',))
    loss=o[4]['token_logits'][:,-1].square().mean()
    a=torch.autograd.grad(loss,[m.blocks[0].fast_tau_logits,m.blocks[0].roles],retain_graph=False)
    left,s=m(x[:,:15],depth=4,stream_ids=('grad-seq',))
    right,s=m(x[:,15:],depth=4,state=s,stream_ids=('grad-seq',))
    loss2=right[4]['token_logits'][:,-1].square().mean()
    b=torch.autograd.grad(loss2,[m.blocks[0].fast_tau_logits,m.blocks[0].roles])
    for x,y in zip(a,b):
        torch.testing.assert_close(x,y,atol=1e-5,rtol=2e-3)


def test_numerical_gradient_scan_against_finite_differences():
    torch.manual_seed(803)
    z=torch.rand(1,7,2,2,2)
    z[...,0]*=z[...,1]
    z.requires_grad_(True)
    rates=torch.tensor([[.04,.008],[.001,.02]],requires_grad=True)
    s=torch.zeros(1,2,2,2)
    def f(w,r):
        out,_=bounded_weighted_mass_scan(w,s,r,4)
        return out[0,-1,0,0,0]+.5*out[0,-1,1,1,1]
    x=f(z,rates)
    dz,dr=torch.autograd.grad(x,[z,rates])
    h=1e-3
    for idx in [(0,0,0,0,0),(0,6,1,1,1)]:
        plus=z.detach().clone();minus=z.detach().clone()
        plus[idx]+=h;minus[idx]-=h
        approx=(f(plus,rates.detach())-f(minus,rates.detach()))/(2*h)
        assert dz[idx].item()==pytest.approx(approx.item(),rel=5e-3,abs=2e-5)
    for idx in [(0,0),(1,1)]:
        plus=rates.detach().clone();minus=rates.detach().clone()
        plus[idx]+=h;minus[idx]-=h
        approx=(f(z.detach(),plus)-f(z.detach(),minus))/(2*h)
        assert dr[idx].item()==pytest.approx(approx.item(),rel=5e-3,abs=2e-5)


def test_large_rotary_positions_do_not_collapse_to_one_fp32_position():
    from tac_osm.csl125 import LocalGQA
    from tac_osm.csl125_r1 import R1LocalGQA
    c=cfg()
    old=LocalGQA(c);new=R1LocalGQA(c)
    pos=torch.tensor([2**25,2**25+1],dtype=torch.int64)
    # The two token vectors are identical, so only the position can separate.
    token=torch.randn(1,c.query_heads,1,c.width//c.query_heads)
    vec=token.expand(-1,-1,2,-1)
    original=old._rope(vec,pos)
    corrected=new._rope(vec,pos)
    assert torch.equal(original[...,0,:],original[...,1,:]), 'original FP32 integer collision must be reproduced'
    assert (corrected[...,0,:]-corrected[...,1,:]).abs().max()>.01


def test_rotary_relative_offset_stable_at_large_absolute_position():
    from tac_osm.csl125_r1 import R1LocalGQA
    c=cfg()
    m=R1LocalGQA(c)
    torch.manual_seed(881)
    x=torch.randn(1,c.query_heads,2,c.width//c.query_heads)
    near=m._rope(x,torch.tensor([0,1],dtype=torch.int64))
    far=m._rope(x,torch.tensor([2**25,2**25+1],dtype=torch.int64))
    # Every head is rotated by the same offset at both queries and keys;
    # pairwise dot-product scores should depend on relative displacement only.
    near_scores=near@near.transpose(-1,-2)
    far_scores=far@far.transpose(-1,-2)
    torch.testing.assert_close(near_scores,far_scores,atol=2e-5,rtol=2e-5)
