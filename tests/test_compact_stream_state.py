"""Untrained engineering controls; zero evidence of learned capability."""
from dataclasses import fields, is_dataclass, replace
import copy

import pytest
import torch
import numpy as np

from research.compact_state_20261108.reference import copy_values, copy_pullback

from tac_osm.csl001 import CSLState, LayerState
from tac_osm.csl125 import CSL125State, LocalKVState
from tac_osm.csl125_r1 import R1LayerState
from tac_osm.csl125_r2 import CSL125R2, CSL125R2Config
from tac_osm.compact_stream_state import compact_stream_state, stream_state_storage


def leaves(node):
    # Independent test traversal, not implementation helper.
    if isinstance(node, torch.Tensor):
        yield node
    elif is_dataclass(node):
        for f in fields(node):
            yield from leaves(getattr(node, f.name))
    elif isinstance(node, (tuple, list)):
        for n in node:
            yield from leaves(n)
    elif isinstance(node, dict):
        for n in node.values():
            yield from leaves(n)


def exact(a, b):
    assert type(a) is type(b)
    if isinstance(a, torch.Tensor):
        assert a.shape == b.shape and a.dtype == b.dtype and a.device == b.device
        assert torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8))
    elif is_dataclass(a):
        for f in fields(a):
            exact(getattr(a, f.name), getattr(b, f.name))
    elif isinstance(a, (tuple, list)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            exact(x, y)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for k in a:
            exact(a[k], b[k])
    else:
        assert a == b


def cfg():
    return CSL125R2Config(vocab_size=32, width=16, ff_width=32, layers=4,
        role_count=2, state_width=8, rungs=(2,4), attention_layers=(2,4),
        query_heads=2, kv_heads=1, attention_window=16)


def test_noncontiguous_aliases_empty_nan_signed_zero_and_metadata():
    owner = torch.tensor([float('nan'), -0., float('inf'), -2., 3., 4.], dtype=torch.float64)
    view = owner[::2]
    layer = R1LayerState(view, view, owner[:0], owner[1:3], owner[2:4])
    state = CSL125State((layer,), (LocalKVState(view, view, 129),), ('entity',))
    rng = torch.get_rng_state().clone()
    small = compact_stream_state(state)
    exact(state, small)
    exact(small, compact_stream_state(small))
    assert torch.equal(rng, torch.get_rng_state())
    for old, new in zip(leaves(state), leaves(small)):
        reference=copy_values([old.detach().numpy()])[0]
        assert new.detach().numpy().tobytes()==reference.tobytes()
        assert not np.shares_memory(reference,old.detach().numpy())
        assert new.is_contiguous()
        assert new.untyped_storage().nbytes() == new.numel() * new.element_size()
        if new.numel():
            assert old.untyped_storage().data_ptr() != new.untyped_storage().data_ptr()
    ptrs = [t.untyped_storage().data_ptr() for t in leaves(small) if t.numel()]
    assert len(ptrs) == len(set(ptrs))
    small.layers[0].fast.fill_(9.)
    assert torch.isnan(state.layers[0].fast[0])
    assert torch.isnan(small.layers[0].slow[0])


def test_detach_alone_retains_storage_and_copy_identity_jacobian():
    source = torch.arange(100., requires_grad=True)
    state = CSLState((LayerState(source[-2:], source[-3:], source[-4:]),), ('x',))
    assert stream_state_storage(state.detached())['retained_storage_bytes'] == 400
    compact = compact_stream_state(state)
    report = stream_state_storage(compact)
    assert report['retained_storage_bytes'] == report['logical_bytes'] == 36
    assert report['requires_grad_tensors'] == 3
    torch.autograd.backward(list(leaves(compact)), [torch.ones_like(t) for t in leaves(compact)])
    reference = torch.zeros_like(source); reference[-2:] += 1; reference[-3:] += 1; reference[-4:] += 1
    assert torch.equal(source.grad, reference)
    pullback=copy_pullback([np.ones(2),np.ones(3),np.ones(4)])
    independent=np.zeros(100)
    for length,grad in zip((2,3,4),pullback):
        independent[-length:]+=grad
    assert np.array_equal(source.grad.numpy(),independent)


@pytest.mark.parametrize('seed', [11,22,33])
@pytest.mark.parametrize('depth', [2,4])
def test_original_model_all_heads_streaming_continuation(seed, depth):
    torch.manual_seed(seed); torch.set_num_threads(1)
    model = CSL125R2(cfg()).eval()
    tokens = torch.randint(0,32,(2,90)); ids=('a','b')
    with torch.no_grad():
        _, state = model(tokens[:,:65], depth=depth, stream_ids=ids)
        before = stream_state_storage(state)
        compact = compact_stream_state(state)
        exact(state, compact)
        after = stream_state_storage(compact)
        assert after['retained_storage_bytes'] == after['logical_bytes']
        assert after['retained_storage_bytes'] < before['retained_storage_bytes']
        assert stream_state_storage(state.detached()) == before
        for start,end in ((65,66),(66,73),(73,90)):
            out,state=model(tokens[:,start:end],depth=depth,state=state,stream_ids=ids,all_rungs=True)
            copied,new=model(tokens[:,start:end],depth=depth,state=compact,stream_ids=ids,all_rungs=True)
            exact(out,copied); exact(state,new)
            compact=compact_stream_state(new)
        with pytest.raises(ValueError,match='stream'):
            model(tokens[:,:1],depth=depth,state=compact,stream_ids=('wrong','b'))
        with pytest.raises(ValueError,match='depth'):
            model(tokens[:,:1],depth=6-depth,state=compact,stream_ids=ids)


@pytest.mark.parametrize('seed', [11,22,33])
def test_attached_full_model_and_incoming_state_gradients_are_preserved(seed):
    torch.manual_seed(seed); torch.set_num_threads(1)
    base=CSL125R2(cfg()).eval(); other=copy.deepcopy(base)
    tokens=torch.randint(0,32,(1,13))
    def execute(model, do_copy):
        _,state=model(tokens[:,:5],depth=4,stream_ids=('x',))
        incoming=list(leaves(state))
        for t in incoming:
            t.retain_grad()
        if do_copy:
            state=compact_stream_state(state)
        outputs,_=model(tokens[:,5:],depth=4,state=state,stream_ids=('x',),all_rungs=True)
        loss=sum(t.double().square().mean() for t in leaves(outputs))
        loss.backward()
        return outputs, [t.grad for t in incoming]
    out,original=execute(base,False); got,copied=execute(other,True)
    exact(out,got)
    for a,b in zip(original,copied):
        assert a is not None and b is not None
        torch.testing.assert_close(a,b,atol=5e-6,rtol=5e-5)
    for (name,a),(other_name,b) in zip(base.named_parameters(),other.named_parameters()):
        assert name == other_name and torch.equal(a,b)
        assert a.grad is not None and b.grad is not None, name
        assert torch.isfinite(a.grad).all() and torch.isfinite(b.grad).all(), name
        torch.testing.assert_close(a.grad,b.grad,atol=5e-6,rtol=5e-5)


def test_unknown_state_subtypes_and_unsupported_layout_fail_closed():
    class FutureState(CSL125State):
        pass
    with pytest.raises(TypeError,match='top-level'):
        compact_stream_state(FutureState((),(),None))
    for t in (torch.empty(2,device='meta'), torch.ones(2).to_sparse(),
              torch.quantize_per_tensor(torch.ones(2),.1,0,torch.quint8)):
        state=CSLState((LayerState(t,t,t),),None)
        with pytest.raises(TypeError,match='materialized dense'):
            compact_stream_state(state)


def test_future_poison_and_source_parameters_rng_unchanged():
    torch.manual_seed(23); torch.set_num_threads(1)
    m=CSL125R2(cfg()).eval()
    tokens=torch.randint(0,32,(1,17)); poisoned=tokens.clone();poisoned[:,11:]=(poisoned[:,11:]+1)%32
    parameters={n:p.detach().clone() for n,p in m.named_parameters()}
    with torch.no_grad():
        a,state=m(tokens,depth=4,stream_ids=('source',),all_rungs=True)
        b,_=m(poisoned,depth=4,stream_ids=('source',),all_rungs=True)
        for depth in a:
            # Last-position-only heads are legitimately after the poison boundary.
            for name in ('token_logits','field_logits'):
                exact(a[depth][name][:,:11],b[depth][name][:,:11])
        rng=torch.get_rng_state().clone()
        compact_stream_state(state)
        assert torch.equal(rng,torch.get_rng_state())
    assert all(torch.equal(parameters[n],p) for n,p in m.named_parameters())
