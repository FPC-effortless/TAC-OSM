"""Untrained source-integrity controls, not checkpoint or learned-quality data."""
from copy import deepcopy
from dataclasses import asdict, fields, is_dataclass
import hashlib
import json
import struct

import pytest
import torch
from tac_osm.csl125_r3 import CSL125R3Config
from tac_osm.csl125_tool_model import CSL125R4
from tac_osm.csl125_memory_precision import enable_precise_memory_
from tac_osm.compact_stream_state import compact_stream_state
from tac_osm.state_ingress import validate_restored_stream_state
from tac_osm.stream_binding import (capture_stream_binding,binding_sha256,
    verify_stream_binding,tensor_fingerprint,StreamBindingError)


def make(seed=11,precise=False):
    torch.set_num_threads(1);torch.manual_seed(seed)
    cfg=CSL125R3Config(vocab_size=256,width=32,ff_width=64,layers=4,
        role_count=4,state_width=8,rungs=(2,4),attention_layers=(2,4),
        query_heads=4,kv_heads=2,attention_window=16,correction_rank=4)
    model=CSL125R4(cfg).eval()
    return enable_precise_memory_(model) if precise else model


def context(depth=4,length=17):
    return dict(depth=depth,stream_ids=('owner',),observed_tokens=length)


def prefix(model,depth=4,length=17):
    s=model.initial_state(1,torch.device('cpu'),torch.float32,depth=depth,stream_ids=('owner',))
    if length:
        _,s=model(torch.arange(length).reshape(1,-1).remainder(256),depth=depth,state=s,stream_ids=('owner',))
    return s


def verify(m,s,b,ctx):
    return verify_stream_binding(m,s,b,expected_binding_sha256=binding_sha256(b),**ctx)


def leaves(x):
    if isinstance(x,torch.Tensor):yield x
    elif is_dataclass(x):
        for f in fields(x):yield from leaves(getattr(x,f.name))
    elif isinstance(x,(tuple,list)):
        for v in x:yield from leaves(v)
    elif isinstance(x,dict):
        for v in x.values():yield from leaves(v)


def test_ingress_alone_accepts_other_seed_but_pinned_binding_rejects():
    a=make(11);b=make(22);ctx=context()
    with torch.no_grad():
        s=prefix(a)
        assert validate_restored_stream_state(b,s,**ctx)['status'].startswith('PASS')
        frozen=capture_stream_binding(a,s,**ctx)
        with pytest.raises(StreamBindingError,match='model_binding_mismatch'):verify(b,s,frozen,ctx)


@pytest.mark.parametrize('seed',[11,22,33])
@pytest.mark.parametrize('precise',[False,True])
@pytest.mark.parametrize('depth',[2,4])
@pytest.mark.parametrize('length',[0,17,65])
def test_actual_owned_copied_model_and_compact_state(seed,precise,depth,length):
    m=make(seed,precise);ctx=context(depth,length)
    with torch.no_grad():
        s=prefix(m,depth,length);weights={n:p.clone() for n,p in m.state_dict().items()}
        identities=[id(t) for t in leaves(s)];values=[t.clone() for t in leaves(s)]
        rng=torch.get_rng_state().clone()
        receipt=capture_stream_binding(m,s,**ctx)
        receipt=json.loads(json.dumps(receipt))
        pin=binding_sha256(receipt)
        for candidate in (s,s.detached(),compact_stream_state(s)):
            assert capture_stream_binding(m,candidate,**ctx)==receipt
            assert verify_stream_binding(m,candidate,receipt,expected_binding_sha256=pin,**ctx)['independently_qualified'] is False
        copied=deepcopy(m);assert verify(copied,compact_stream_state(s),receipt,ctx)['source_authentication']=='NOT_ESTABLISHED'
        x=torch.tensor([[42]])
        a,sa=m(x,depth=depth,state=s,stream_ids=('owner',),all_rungs=True,emit_repair_logits=True,candidate_repair_topk=4)
        b,sb=copied(x,depth=depth,state=compact_stream_state(s),stream_ids=('owner',),all_rungs=True,emit_repair_logits=True,candidate_repair_topk=4)
        assert all(torch.equal(u,v) for u,v in zip(leaves(a),leaves(b)))
        assert all(torch.equal(u,v) for u,v in zip(leaves(sa),leaves(sb)))
        assert identities==[id(t) for t in leaves(s)]
        assert all(torch.equal(u,v) for u,v in zip(values,leaves(s)))
        assert all(torch.equal(weights[n],p) for n,p in m.state_dict().items())
        assert torch.equal(rng,torch.get_rng_state())


@pytest.mark.parametrize('fault',['parameter','buffer','config','mode','training','state'])
def test_named_model_and_valid_state_mutations_cannot_restore(fault):
    m=make();ctx=context()
    with torch.no_grad():
        s=prefix(m);receipt=capture_stream_binding(m,s,**ctx)
        if fault=='parameter':m.embedding.weight[0,0]+=1
        elif fault=='buffer':m.attention['2'].inv_freq[0]+=1
        elif fault=='config':object.__setattr__(m.config,'max_correction_gain',0.125)
        elif fault=='training':m.train()
        elif fault=='mode':
            enable_precise_memory_(m);s=prefix(m)
        else:s.layers[0].fast.zero_()  # still finite and inside the mass cone
        validate_restored_stream_state(m,s,**ctx)
        with pytest.raises(StreamBindingError,match='state_binding_mismatch' if fault=='state' else 'model_binding_mismatch'):
            verify(m,s,receipt,ctx)


@pytest.mark.parametrize('fault',['missing_pin','bad_pin','edited','unknown_schema','missing_key','owner','depth','cutoff'])
def test_frozen_receipt_pin_schema_and_independent_context(fault):
    m=make();ctx=context()
    with torch.no_grad():s=prefix(m);receipt=capture_stream_binding(m,s,**ctx)
    pin=binding_sha256(receipt);code='binding_context_mismatch'
    if fault=='missing_pin':pin=None;code='trusted_binding_pin_required'
    elif fault=='bad_pin':pin='0'*64;code='binding_hash_mismatch'
    elif fault=='edited':receipt['state_sha256']='f'*64;code='binding_hash_mismatch'
    elif fault=='unknown_schema':receipt['schema']='independently-qualified';pin=binding_sha256(receipt);code='binding_format'
    elif fault=='missing_key':receipt.pop('scope');pin=binding_sha256(receipt);code='binding_format'
    elif fault=='owner':ctx['stream_ids']=('other',)
    elif fault=='depth':ctx['depth']=2
    else:ctx['observed_tokens']=18
    with pytest.raises(StreamBindingError,match='^'+code+'$'):
        verify_stream_binding(m,s,receipt,expected_binding_sha256=pin,**ctx)


def test_independent_binary_and_canonical_hash_reference():
    t=torch.tensor([1.,-0.,1.+2.**-40],dtype=torch.float64)
    record=tensor_fingerprint(t)
    assert record=={'dtype':'torch.float64','shape':[3],
                   'bytes_sha256':hashlib.sha256(struct.pack('=ddd',1.,-0.,1.+2.**-40)).hexdigest()}
    assert tensor_fingerprint(torch.empty(0))['bytes_sha256']==hashlib.sha256(b'').hexdigest()
    assert tensor_fingerprint(torch.tensor(1.))['shape']==[]
    assert binding_sha256({'z':1,'a':'é'})==hashlib.sha256('{"a":"é","z":1}'.encode()).hexdigest()


@pytest.mark.parametrize('seed',[11,22,33])
def test_binding_never_detaches_live_state_or_changes_parameter_gradients(seed):
    a=make(seed,True);b=deepcopy(a);x=torch.randint(0,256,(1,17))
    def run(m,bind):
        _,s=m(x[:,:5],depth=4,stream_ids=('owner',))
        incoming=list(leaves(s))
        for t in incoming:t.retain_grad()
        if bind:
            ctx=context(4,5);frozen=capture_stream_binding(m,s,**ctx);verify(m,s,frozen,ctx)
        out,_=m(x[:,5:],depth=4,state=s,stream_ids=('owner',),all_rungs=True,emit_repair_logits=True)
        sum(t.double().square().mean() for t in leaves(out)).backward()
        return [t.grad for t in incoming]
    ag=run(a,False);bg=run(b,True)
    for u,v in zip(ag,bg):
        assert u is not None and v is not None
        assert torch.isfinite(u).all() and torch.isfinite(v).all()
        torch.testing.assert_close(u,v,atol=5e-6,rtol=5e-5)
    for (name,p),(other,q) in zip(a.named_parameters(),b.named_parameters()):
        assert name==other and p.grad is not None and q.grad is not None
        assert torch.isfinite(p.grad).all() and torch.isfinite(q.grad).all()
        torch.testing.assert_close(p.grad,q.grad,atol=5e-6,rtol=5e-5)
