"""Constructed no-fit graph checks; zero trained capability evidence."""
from dataclasses import replace
import pytest
import torch
from .models import AdaptiveRepresentationModel,FrozenBPE,MODES
from .reference import forward as reference_forward

torch.set_num_threads(2)

@pytest.fixture(scope='module')
def tokenizer():
    # This is a fixture tokenizer, not an authenticated real-source training fit.
    return FrozenBPE.fit([b'alpha beta gamma delta '*32,b'zero one two three four '*32],partition='FIT')

def model(name,tokenizer):
    torch.manual_seed(101006)
    return AdaptiveRepresentationModel(name,tokenizer=tokenizer if name=='a5_fixed_bpe' else None).eval()

@pytest.mark.parametrize('name',MODES)
def test_live_active_size_band(name,tokenizer):
    m=model(name,tokenizer);n=m.counts()
    assert 95000<=n['trainable']==n['deployed']<=105000
    assert abs(n['trainable']-100370)/max(n['trainable'],100370)<.01

@pytest.mark.parametrize('name',MODES)
def test_future_poison_preserves_earlier_logits(name,tokenizer):
    m=model(name,tokenizer);x=torch.arange(12)[None];z=x.clone();z[:,7:]=255
    with torch.inference_mode():a,_=m(x,owners=('a',));b,_=m(z,owners=('a',))
    assert torch.equal(a[:,:7],b[:,:7])

@pytest.mark.parametrize('name',MODES)
def test_whole_chunk_serial_equivalence(name,tokenizer):
    m=model(name,tokenizer);x=torch.tensor([[4,5,6,7,8,9,10,11]])
    with torch.inference_mode():
        full,s=m(x,owners=('a',));left,t=m(x[:,:3],owners=('a',));right,u=m(x[:,3:],state=t,owners=('a',))
        state=None;outs=[]
        for t in range(x.shape[1]):
            o,state=m(x[:,t:t+1],state=state,owners=('a',));outs.append(o)
    assert torch.equal(full,torch.cat((left,right),1))
    assert torch.equal(full,torch.cat(outs,1))
    assert u.clock==s.clock and u.pending_bytes==s.pending_bytes
    for k in u.tensors:assert torch.equal(u.tensors[k],s.tensors[k])

@pytest.mark.parametrize('name',MODES)
def test_owner_rebinding_rejected_and_entity_isolation(name,tokenizer):
    m=model(name,tokenizer);x=torch.arange(16).reshape(2,8)
    with torch.inference_mode():
        output,s=m(x,owners=('a','b'))
        for i,owner in enumerate(('a','b')):
            single,_=m(x[i:i+1],owners=(owner,))
            assert torch.equal(output[i:i+1],single)
        with pytest.raises(ValueError):m(x,state=s,owners=('b','a'))

@pytest.mark.parametrize('name',MODES)
def test_all_parameter_tensors_receive_finite_graph_gradients(name,tokenizer):
    m=model(name,tokenizer);x=torch.arange(32).reshape(2,16)%256
    if name=='a5_adaptive':
        # Coverage is across legal route interventions, not a claim an unfitted
        # gate already visits all branches or learns a good policy.
        outputs=[m(x,owners=('a','b'),force=force)[0] for force in ('byte','discrete','latent')]
        loss=sum(y.square().mean() for y in outputs)
        # Unforced selector loss explicitly exercises controller parameters.
        loss+=m(x,owners=('a','b'))[0].square().mean()
    else:loss=m(x,owners=('a','b'))[0].square().mean()
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() and bool(p.grad.abs().sum()>0) for p in m.parameters())

def test_selected_discrete_identity_is_load_bearing_without_raw_byte_head(tokenizer):
    m=model('a5_discrete',tokenizer);x=torch.tensor([[1,2,3,4,5,6,7,8]]);z=255-x
    with torch.inference_mode():
        a,_=m(x,owners=('a',),code_override=2)
        b,_=m(z,owners=('a',),code_override=2)
        c,_=m(x,owners=('a',),code_override=5)
        assert torch.equal(a,b),'raw bytes bypassed the fixed selected code'
        assert not torch.equal(a[:,3:],c[:,3:]),'selected code failed to affect predictions'
        old=a.clone();m.embedding.weight.normal_(5,3)
        d,_=m(x,owners=('a',),code_override=2)
        assert torch.equal(old,d),'raw byte embedding leaked around committed code ID'

def test_controller_forced_discrete_has_same_no_bypass_property(tokenizer):
    m=model('a5_adaptive',tokenizer);x=torch.arange(8)[None]
    with torch.inference_mode():
        a,_=m(x,owners=('a',),force='discrete',code_override=4)
        b,_=m(255-x,owners=('a',),force='discrete',code_override=4)
        assert torch.equal(a,b)

@pytest.mark.parametrize('force',('byte','discrete','latent',None))
def test_sparse_inference_and_training_surrogate_have_equal_forward(force,tokenizer):
    m=model('a5_adaptive',tokenizer);x=torch.arange(16)[None]
    with torch.inference_mode():
        a,s=m(x,owners=('a',),force=force,execution='inference');r1=dict(m.last_receipt)
        b,t=m(x,owners=('a',),force=force,execution='surrogate');r2=dict(m.last_receipt)
    assert torch.equal(a,b)
    assert all(torch.equal(s.tensors[k],t.tensors[k]) for k in s.tensors)
    assert r2['candidate_recurrent_calls']>=r1['candidate_recurrent_calls']

@pytest.mark.parametrize('name',('a5_discrete','a5_latent'))
def test_pending_bytes_do_not_reach_decoder_before_commit(name,tokenizer):
    m=model(name,tokenizer);x=torch.tensor([[1,2,3,4,5,6,7,8]])
    with torch.inference_mode():
        y,_=m(x,owners=('a',));r=m.last_receipt
    assert torch.equal(y[:,0],y[:,1]) and torch.equal(y[:,1],y[:,2])
    assert torch.equal(y[:,3],y[:,4]) and torch.equal(y[:,4],y[:,5])
    assert r['recurrent_updates']==2 and r['source_bytes']==8
    assert [e['end'] for e in r['events']]==[4,8]

def test_bpe_is_fit_only_lossless_and_prefix_recomputation_declared(tokenizer):
    for raw in (bytes(range(256)),b'alpha beta alpha beta',b'new_rare_identifier_123'):
        assert tokenizer.decode(tokenizer.encode(raw))==raw
    with pytest.raises(ValueError):FrozenBPE.fit([b'abc '*100],partition='DEV')
    m=model('a5_fixed_bpe',tokenizer)
    with torch.inference_mode():m(torch.arange(8)[None],owners=('a',))
    assert m.last_receipt['prefix_retokenization']
    assert m.last_receipt['raw_history_bytes']==8
    assert m.last_receipt['prefix_hidden_reuse']
    assert m.last_receipt['recurrent_updates']==9
    assert m.last_receipt['reused_prefix_hidden_states']>0
    assert FrozenBPE.from_receipt(tokenizer.receipt()).receipt()==tokenizer.receipt()

def test_invalid_or_foreign_state_and_unknown_precision_rejected(tokenizer):
    m=model('a5_latent',tokenizer);s=m.start(('a',))
    with pytest.raises(ValueError):m(torch.tensor([[256]]),state=s,owners=('a',))
    with pytest.raises(ValueError):m(torch.tensor([[2.]]),state=s,owners=('a',))
    with pytest.raises(ValueError):m(torch.tensor([[2]]),state=replace(s,clock=-1),owners=('a',))
    with pytest.raises(ValueError):m(torch.tensor([[2]]),state=replace(s,tensors={'hidden':s.tensors['hidden']}),owners=('a',))

@pytest.mark.parametrize('name',MODES)
def test_full_reference_output_state_and_parameter_gradients(name,tokenizer):
    m=model(name,tokenizer).double();x=torch.arange(16).reshape(2,8)
    # Fixed route exercises discrete identity in the adaptive parameter graph;
    # natural controller gradients are additionally compared without override.
    for force in (('discrete',None) if name=='a5_adaptive' else (None,)):
        m.zero_grad(set_to_none=True)
        p={k:v.detach().clone().requires_grad_(True) for k,v in m.named_parameters()}
        a,s=m(x,owners=('a','b'),force=force)
        b,t=reference_forward(p,name,x,owners=('a','b'),merges=tokenizer.merges,force=force)
        assert torch.allclose(a,b,atol=1e-12,rtol=1e-12)
        assert s.pending_bytes==t.pending_bytes and s.pending_modes==t.pending_modes
        for k in s.tensors:assert torch.allclose(s.tensors[k],t.tensors[k],atol=1e-12,rtol=1e-12)
        weights=torch.linspace(-1,1,a.numel(),dtype=torch.float64).reshape_as(a)
        loss_a=(a*weights).sum();loss_b=(b*weights).sum()
        if name=='a5_fixed_bpe':
            loss_a+=s.tensors['bpe_cache'].square().sum()*.01
            loss_b+=t.tensors['bpe_cache'].square().sum()*.01
        loss_a.backward();loss_b.backward()
        for k,v in m.named_parameters():
            if v.grad is None:assert p[k].grad is None
            else:assert torch.allclose(v.grad,p[k].grad,atol=1e-10,rtol=1e-9),k

def test_bpe_cache_invalidates_after_parameter_change_and_model_reload(tokenizer):
    m=model('a5_fixed_bpe',tokenizer).double();x=torch.tensor([list(b'alpha beta gamma')])
    with torch.inference_mode():
        _,s=m(x[:,:8],owners=('a',))
        m.gru.weight_hh.add_(.01)
        after,t=m(x[:,8:],state=s,owners=('a',))
        full,u=m(x,owners=('a',))
        assert torch.allclose(after,full[:,8:],atol=1e-12,rtol=1e-12)
        assert not m.last_receipt['cache_guard_valid_on_entry']
        other=model('a5_fixed_bpe',tokenizer).double();other.load_state_dict(m.state_dict())
        restored,_=other(x[:,8:],state=s,owners=('a',))
        assert torch.allclose(restored,after,atol=1e-12,rtol=1e-12)

def test_bpe_revision_cache_matches_naive_reference_on_realistic_merges(tokenizer):
    m=model('a5_fixed_bpe',tokenizer).double();x=torch.tensor([list(b'alpha beta alpha beta')])
    with torch.inference_mode():
        y,s=m(x,owners=('a',));p={k:v.clone() for k,v in m.named_parameters()}
        z,t=reference_forward(p,'a5_fixed_bpe',x,owners=('a',),merges=tokenizer.merges)
    assert torch.allclose(y,z,atol=1e-12,rtol=1e-12)
    assert s.cache_tokens==t.cache_tokens
    assert torch.allclose(s.tensors['bpe_cache'],t.tensors['bpe_cache'],atol=1e-12,rtol=1e-12)

