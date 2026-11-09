"""Source-pinned, untrained integration controls; not model capability evidence."""
from copy import deepcopy
from dataclasses import fields, is_dataclass, replace

import numpy as np
import pytest
import torch

from tac_osm.compact_stream_state import compact_stream_state, stream_state_storage
from tac_osm.csl125 import CSL125State
from tac_osm.csl125_memory_precision import PreciseMemoryState, enable_precise_memory_
from tac_osm.csl125_r3 import CSL125R3Config
from tac_osm.csl125_tool_model import CSL125R4
from research.compact_state_20261108.reference import copy_values


def leaves(value):
    if isinstance(value,torch.Tensor):
        yield value
    elif is_dataclass(value):
        for f in fields(value):
            yield from leaves(getattr(value,f.name))
    elif isinstance(value,(tuple,list)):
        for v in value:
            yield from leaves(v)
    elif isinstance(value,dict):
        for v in value.values():
            yield from leaves(v)


def same(a,b):
    assert type(a) is type(b)
    if isinstance(a,torch.Tensor):
        assert (a.shape,a.dtype,a.device)==(b.shape,b.dtype,b.device)
        assert torch.equal(a.contiguous().view(torch.uint8),b.contiguous().view(torch.uint8))
    elif is_dataclass(a):
        for f in fields(a):
            same(getattr(a,f.name),getattr(b,f.name))
    elif isinstance(a,(tuple,list)):
        assert len(a)==len(b)
        for x,y in zip(a,b):
            same(x,y)
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for k in a:
            same(a[k],b[k])
    else:
        assert a==b


def cfg():
    return CSL125R3Config(vocab_size=256,width=32,ff_width=64,layers=4,
        role_count=4,state_width=8,rungs=(2,4),attention_layers=(2,4),
        query_heads=4,kv_heads=2,attention_window=16,correction_rank=4)


def model(seed,precise=True):
    torch.manual_seed(seed);torch.set_num_threads(1)
    m=CSL125R4(cfg()).eval()
    return enable_precise_memory_(m) if precise else m


def all_heads(m,x,depth,state=None):
    return m(x,depth=depth,state=state,stream_ids=('entity',),all_rungs=True,
             emit_repair_logits=True,candidate_repair_topk=4)


def check_precision(state):
    for s in state.layers:
        assert type(s) is PreciseMemoryState
        assert all(getattr(s,n).dtype==torch.float64 for n in ('fast','slow','fast_mass','slow_mass'))


def test_precise_version_and_precision_are_preserved():
    backing=torch.tensor([1.+2.**-40, -2.**-50, 1.+2.**-39, 2.**-60],dtype=torch.float64)
    v=backing[::2]
    state=CSL125State((PreciseMemoryState(v,v,torch.ones(0,dtype=torch.bfloat16),v,v),),(None,),('one',))
    compact=compact_stream_state(state)
    same(state,compact);check_precision(compact)
    assert not torch.equal(v,v.float().double())
    for original,copied in zip(leaves(state),leaves(compact)):
        assert original.dtype==copied.dtype
        if copied.dtype==torch.float64:
            reference=copy_values([original.numpy()])[0]
            assert copied.numpy().tobytes()==reference.tobytes()
            assert original.untyped_storage().data_ptr()!=copied.untyped_storage().data_ptr()
    report=stream_state_storage(compact)
    assert report['logical_bytes']==report['retained_storage_bytes']


@pytest.mark.parametrize('seed',[61,62,63])
@pytest.mark.parametrize('depth',[2,4])
@pytest.mark.parametrize('precise',[False,True])
def test_r4_all_heads_multiple_chunks_same_partition(seed,depth,precise):
    m=model(seed,precise);tokens=torch.randint(0,256,(1,46))
    with torch.no_grad():
        _,state=all_heads(m,tokens[:,:33],depth)
        before=stream_state_storage(state)
        compact=compact_stream_state(state)
        assert stream_state_storage(compact)['retained_storage_bytes']==before['logical_bytes']
        assert stream_state_storage(compact)['retained_storage_bytes']<before['retained_storage_bytes']
        same(state,compact)
        for start,end in ((33,34),(34,39),(39,46)):
            a,state=all_heads(m,tokens[:,start:end],depth,state)
            b,new=all_heads(m,tokens[:,start:end],depth,compact)
            same(a,b);same(state,new)
            compact=compact_stream_state(new)
            if precise:
                check_precision(compact)
        with pytest.raises(ValueError,match='stream'):
            m(tokens[:,:1],depth=depth,state=compact,stream_ids=('other',))
        with pytest.raises(ValueError,match='depth'):
            m(tokens[:,:1],depth=6-depth,state=compact,stream_ids=('entity',))


@pytest.mark.parametrize('seed',[61,62,63])
def test_all_r4_parameter_and_incoming_state_gradients_preserved(seed):
    a=model(seed);b=deepcopy(a);x=torch.randint(0,256,(1,17))
    def execute(m,copy):
        _,state=all_heads(m,x[:,:5],4)
        incoming=list(leaves(state))
        for t in incoming:
            t.retain_grad()
        if copy:
            state=compact_stream_state(state)
        outs,_=all_heads(m,x[:,5:],4,state)
        sum(t.double().square().mean() for t in leaves(outs) if t.is_floating_point()).backward()
        return outs,[t.grad for t in incoming]
    out,original=execute(a,False);got,copied=execute(b,True);same(out,got)
    for x,y in zip(original,copied):
        assert x is not None and y is not None
        torch.testing.assert_close(x,y,atol=5e-6,rtol=5e-5)
    for (name,p),(name2,q) in zip(a.named_parameters(),b.named_parameters()):
        assert name==name2 and torch.equal(p,q)
        assert p.grad is not None and q.grad is not None,name
        assert torch.isfinite(p.grad).all() and torch.isfinite(q.grad).all(),name
        torch.testing.assert_close(p.grad,q.grad,atol=5e-6,rtol=5e-5)


def test_copy_does_not_upgrade_old_fp32_carry_or_unknown_precise_subclasses():
    m=model(65,False);x=torch.randint(0,256,(1,5))
    with torch.no_grad():
        _,old=m(x,depth=4,stream_ids=('entity',))
        enable_precise_memory_(m)
        old_copy=compact_stream_state(old)
        with pytest.raises(ValueError,match='reset/replay'):
            m(x,depth=4,state=old_copy,stream_ids=('entity',))
        _,state=m(x,depth=4,stream_ids=('entity',))
        class FuturePrecision(PreciseMemoryState):
            pass
        s=state.layers[0]
        unknown=FuturePrecision(s.fast,s.slow,s.conv_tail,s.fast_mass,s.slow_mass)
        with pytest.raises(TypeError,match='unregistered'):
            compact_stream_state(replace(state,layers=(unknown,)+state.layers[1:]))


def test_cpu_autocast_keeps_precision_boundary_and_future_poison():
    m=model(64);x=torch.randint(0,256,(1,17));poison=x.clone();poison[:,11:]=(poison[:,11:]+17)%256
    with torch.no_grad():
        out,_=all_heads(m,x,4);altered,_=all_heads(m,poison,4)
        for depth in out:
            for key in ('token_logits','field_logits','repair_logits'):
                same(out[depth][key][:,:11],altered[depth][key][:,:11])
    with torch.no_grad(),torch.autocast('cpu',dtype=torch.bfloat16):
        _,state=all_heads(m,x[:,:11],4)
        compact=compact_stream_state(state);check_precision(compact);same(state,compact)
        rng=torch.get_rng_state().clone()
        a,_=all_heads(m,x[:,11:12],4,state);b,_=all_heads(m,x[:,11:12],4,compact)
        same(a,b)
        assert all(torch.isfinite(t).all() for t in leaves(a))
        assert torch.equal(rng,torch.get_rng_state())
