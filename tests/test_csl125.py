import pytest
import torch
from tac_osm.csl125 import CSL125, CSL125Config, CAP

@pytest.fixture
def small():
    torch.manual_seed(2026)
    cfg=CSL125Config(vocab_size=256,width=24,layers=4,ff_width=48,
                     role_count=3,state_width=8,rungs=(2,4),attention_layers=(2,4),
                     query_heads=4,kv_heads=2,attention_window=8)
    return CSL125(cfg).eval()

def test_default_125m_limit_and_exact_budget():
    with torch.device('meta'):
        m=CSL125(CSL125Config(),device='meta')
    assert m.parameter_count()==123699534
    assert sum(m.parameter_breakdown().values())==m.parameter_count()
    assert m.parameter_count()<=CAP
    assert m.parameter_breakdown()['embedding_and_tied_lm_head']==32768*768

def test_oversized_rejected_without_allocating():
    with torch.device('meta'):
        with pytest.raises(ValueError,match='ceiling exceeded'):
            CSL125(CSL125Config(ff_width=1536),device='meta')

def test_depths_shapes_no_dup_shared_head(small):
    x=torch.randint(0,256,(2,9))
    out,_=small(x,depth=2)
    assert out[2]['token_logits'].shape==(2,9,256)
    out,_=small(x,all_rungs=True)
    assert set(out)=={2,4}
    assert out[4]['tool_logits'].shape==(2,17)

def test_future_mutation_no_predecision_effect(small):
    x=torch.randint(0,256,(1,9)); y=x.clone();y[:,6:]=torch.randint(0,256,(1,3))
    a,_=small(x,depth=4);b,_=small(y,depth=4)
    torch.testing.assert_close(a[4]['token_logits'][:,:6],b[4]['token_logits'][:,:6],atol=1e-5,rtol=1e-5)

def test_chunked_vs_one_shot_exact_state(small):
    x=torch.randint(0,256,(1,17)); full,fullstate=small(x,depth=4,stream_ids=('a',))
    first,state=small(x[:,:7],depth=4,stream_ids=('a',))
    second,state=small(x[:,7:],depth=4,stream_ids=('a',),state=state)
    torch.testing.assert_close(torch.cat([first[4]['token_logits'],second[4]['token_logits']],dim=1),
                               full[4]['token_logits'],atol=4e-5,rtol=4e-5)
    for a,b in zip(state.layers,fullstate.layers):
        torch.testing.assert_close(a.fast,b.fast,atol=4e-5,rtol=4e-5)
        torch.testing.assert_close(a.slow,b.slow,atol=4e-5,rtol=4e-5)
    for a,b in zip(state.attention,fullstate.attention):
        if a is not None:
            torch.testing.assert_close(a.k,b.k,atol=4e-5,rtol=4e-5)
            assert a.k.shape[-2]<=small.config.attention_window-1

def test_ids_cannot_cross_entity(small):
    x=torch.randint(0,256,(1,5));_,s=small(x,depth=2,stream_ids=('first',))
    with pytest.raises(ValueError,match='stream'):
        small(x,depth=2,state=s,stream_ids=('second',))

def test_depth_change_requires_replay(small):
    x=torch.randint(0,256,(1,5));_,s=small(x,depth=2,stream_ids=('first',))
    with pytest.raises(ValueError,match='depth'):
        small(x,depth=4,state=s,stream_ids=('first',))

def test_route_without_large_vocabulary_projection(small):
    x=torch.randint(0,256,(1,5));out,_=small(x,depth=4,emit_token_logits=False)
    assert 'token_logits' not in out[4]
    assert 'tool_logits' in out[4]

def test_trainable_operators_gradient_connected(small):
    x=torch.randint(0,256,(1,7));out,_=small(x,depth=4)
    loss=out[4]['token_logits'].square().mean()
    loss.backward()
    grads=[small.blocks[0].key.weight.grad,small.attention['2'].q.weight.grad,
           small.embedding.weight.grad]
    assert all(g is not None and torch.isfinite(g).all() for g in grads)

def test_nested_next_token_loss_produces_gradients(small):
    x=torch.randint(0,256,(1,7));loss=small.nested_next_token_loss(x,{2:1,4:1});loss.backward()
    assert torch.isfinite(loss)
    assert small.blocks[-1].up.weight.grad is not None

def test_local_window_bounds(small):
    x=torch.randint(0,256,(1,25));_,s=small(x,depth=4,stream_ids=('c',))
    for a in s.attention:
        if a is not None:
            assert a.k.shape[2]==small.config.attention_window-1
            assert a.position==25

def test_invalid_vocab_token_rejected(small):
    with pytest.raises((IndexError, RuntimeError)):
        small(torch.tensor([[256]],dtype=torch.long),depth=2)

def test_config_invalid_attention_geometry():
    with pytest.raises(ValueError):
        CSL125Config(width=24,layers=4,rungs=(2,4),query_heads=5,attention_layers=(4,))