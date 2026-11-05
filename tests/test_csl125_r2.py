"""No-fit R2 selective-admission tests, not scientific learning qualification."""
import inspect
import math
import numpy as np
import pytest
import torch
from tac_osm.csl125_r1 import CSL125R1, CSL125R1Config, bounded_weighted_mass_scan
from tac_osm.csl125_r2 import (
    CSL125R2, CSL125R2Config, R2RoleStateBlock, selective_affine_scan,
)
from tac_osm.csl125_r2_math_reference import selective_serial


def small_cfg():
    return CSL125R2Config(vocab_size=64, width=32, ff_width=64, layers=4,
        role_count=4, state_width=8, rungs=(2,4), attention_layers=(2,4),
        query_heads=4, kv_heads=2, attention_window=16)


def draw(seed=33, t=137):
    gen=np.random.default_rng(seed)
    a=gen.dirichlet(np.ones(4),size=(2,t)).astype(np.float64)
    v=gen.uniform(-1,1,size=(2,t,8))
    w=np.stack([a[...,None]*v[:,:,None,:],np.broadcast_to(a[...,None],(2,t,4,8))],axis=-1)
    carry=np.zeros((2,4,8,2),dtype='float64')
    rate=gen.uniform(1/32768,0.49,size=(4,8))
    gates=gen.uniform(0,1,size=(2,t,4))
    return w,carry,rate,gates

@pytest.mark.parametrize('size',[1,7,32,64])
def test_stable_affine_scan_vs_independent_float64_serial(size):
    w,carry,rate,gates=draw()
    ref,ref_end=selective_serial(w,carry,rate,gates)
    got,end=selective_affine_scan(torch.tensor(w,dtype=torch.float32),
        torch.tensor(carry,dtype=torch.float32),torch.tensor(rate,dtype=torch.float32),
        torch.tensor(gates,dtype=torch.float32),size)
    assert torch.allclose(got,torch.tensor(ref,dtype=torch.float32),atol=5e-6,rtol=5e-6)
    assert torch.allclose(end,torch.tensor(ref_end,dtype=torch.float32),atol=5e-6,rtol=5e-6)
    assert (got[...,1]>=0).all()
    assert (got[...,0].abs()<=got[...,1]+1e-6).all()


def test_gates_off_hold_exact_memory_state_and_gates_on_recover_r1():
    w,carry,rate,_=draw(t=25)
    start=np.random.default_rng(12).uniform(-.5,.5,size=carry.shape).astype('float32')
    # Starting numerator/mass need not be physically legal for hold invariance.
    inp=torch.tensor(w,dtype=torch.float32)
    initial=torch.tensor(start,dtype=torch.float32)
    rr=torch.tensor(rate,dtype=torch.float32)
    off=torch.zeros((2,25,4),dtype=torch.float32)
    zero,held=selective_affine_scan(inp,initial,rr,off,7)
    assert torch.equal(held,initial)
    assert torch.equal(zero,initial[:,None].expand_as(zero))
    on=torch.ones_like(off)
    fixed,post=selective_affine_scan(inp,initial,rr,on,7)
    legacy,legacy_post=bounded_weighted_mass_scan(inp,initial,rr,7)
    assert torch.allclose(fixed,legacy,atol=2e-6,rtol=2e-6)
    assert torch.allclose(post,legacy_post,atol=2e-6,rtol=2e-6)


def test_no_future_source_path_and_learned_gate_gradient_connectivity():
    torch.manual_seed(31)
    m=CSL125R2(small_cfg()).eval()
    a=torch.randint(0,64,(1,16))
    b=a.clone();b[:,12:]=torch.randint(0,64,(1,4))
    y,_=m(a,depth=4);z,_=m(b,depth=4)
    assert torch.allclose(y[4]['token_logits'][:,:12],z[4]['token_logits'][:,:12],atol=1e-5)
    loss=m.nested_next_token_loss(a)
    loss.backward()
    gates=[p for n,p in m.named_parameters() if 'write_gate' in n]
    assert len(gates)==8 and all(p.grad is not None for p in gates)
    assert all(torch.isfinite(p.grad).all() for p in gates)
    assert any(p.grad.abs().max()>0 for p in gates)


def test_streaming_matches_single_pass_and_rejects_changed_identity():
    torch.manual_seed(7)
    m=CSL125R2(small_cfg()).eval()
    x=torch.randint(0,64,(1,32))
    with torch.no_grad():
        full,s=m(x,depth=4,stream_ids=('owner',))
        state=None
        chunks=[]
        for begin,end in ((0,7),(7,20),(20,32)):
            outs,state=m(x[:,begin:end],depth=4,state=state,stream_ids=('owner',))
            chunks.append(outs[4]['token_logits'])
        assert torch.allclose(torch.cat(chunks,1),full[4]['token_logits'],atol=2e-5,rtol=2e-5)
        with pytest.raises(ValueError, match='stream'):
            m(x[:,:1],depth=4,state=state,stream_ids=('other',))


def test_zero_gate_can_preserve_memories_while_tokens_continue():
    torch.manual_seed(12)
    c=small_cfg();block=R2RoleStateBlock(c).eval()
    with torch.no_grad():
        block.write_gate.weight.zero_()
        block.write_gate.bias.fill_(80.)
        start=block.initial_state(1,torch.device('cpu'),torch.float32)
        _,state=block(torch.randn(1,7,c.width),start)
        block.write_gate.bias.fill_(-100.)
        _,new=block(torch.randn(1,24,c.width),state)
        for k in ('fast','slow','fast_mass','slow_mass'):
            assert torch.equal(getattr(state,k),getattr(new,k))
        assert not torch.equal(state.conv_tail,new.conv_tail), 'local convolution history must still advance'


def test_meta_full_geometry_stays_below_125m_and_counts_every_gate():
    with torch.device('meta'):
        m=CSL125R2(device='meta')
    assert m.parameter_count()==123_822_574
    assert m.parameter_count()<125_000_000
    assert sum('write_gate' in n and p.numel() or 0 for n,p in m.named_parameters())==123_040
    assert all(p.requires_grad for p in m.parameters())


def test_invalid_gate_data_and_bad_config_fail_closed():
    w,carry,rate,gates=draw(t=5)
    z=torch.tensor(w,dtype=torch.float32);c=torch.tensor(carry,dtype=torch.float32)
    r=torch.tensor(rate,dtype=torch.float32);g=torch.tensor(gates,dtype=torch.float32)
    with pytest.raises(ValueError,match='gate'):
        selective_affine_scan(z,c,r,torch.full_like(g,-0.01),3)
    with pytest.raises(ValueError,match='gate'):
        selective_affine_scan(z,c,r,torch.full_like(g,float('nan')),3)
    with pytest.raises(ValueError,match='rate'):
        selective_affine_scan(z,c,r*0,g,3)
    with pytest.raises(ValueError,match='bias'):
        CSL125R2Config(write_bias_initial=float('nan'))
    assert 'gate_override' not in inspect.signature(CSL125R2.forward).parameters


def test_constant_slot_address_normalized_read_cancels_address_magnitude():
    # Source R1 invariant: stationary per-token role assignments do not
    # establish per-role identity-specific content; R2 retains this limitation
    # when all gates and decay rates match.
    x=np.array([1.,-1.,0.3,0.8])
    probs=np.array([.5,.3,.15,.05])
    N=np.zeros(4);D=np.zeros(4)
    alpha=np.exp(-1/64)
    for v in x:
        N=alpha*N+(1-alpha)*probs*v
        D=alpha*D+(1-alpha)*probs
    assert np.max(N/D)-np.min(N/D)<1e-12


def test_cpu_bf16_autocast_forward_backward_is_finite():
    torch.manual_seed(129)
    model=CSL125R2(small_cfg()).train()
    tokens=torch.randint(0,64,(2,24))
    with torch.autocast('cpu', dtype=torch.bfloat16):
        outputs,_=model(tokens,all_rungs=True)
        loss=sum(torch.nn.functional.cross_entropy(
            outputs[r]['token_logits'][:,:-1].reshape(-1,64), tokens[:,1:].reshape(-1))
            + .01 * outputs[r]['tool_logits'].square().mean()
            + .01 * outputs[r]['field_logits'].square().mean()
            + .01 * outputs[r]['accept_logit'].square().mean()
            for r in model.config.rungs)
    assert torch.isfinite(loss)
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_training_gate_leakage_boundary_excludes_external_oracle():
    # No gate input argument, provider label, reward or future value is accepted
    # by the model. This does not prove source-pipeline absence of leakage.
    torch.manual_seed(130)
    model=CSL125R2(small_cfg()).eval()
    x=torch.randint(0,64,(1,7))
    with pytest.raises(TypeError):
        model(x,depth=2,gate_override=torch.ones(1,7,4))
    with pytest.raises(TypeError):
        model(x,depth=2,target_labels=x)
