"""Constructed original-source controls; zero capability or security claims."""
from copy import deepcopy
from dataclasses import fields, is_dataclass, replace
import numpy as np
import pytest
import torch
from tac_osm.csl125 import CSL125State
from tac_osm.csl125_r3 import CSL125R3Config
from tac_osm.csl125_tool_model import CSL125R4
from tac_osm.csl125_memory_precision import enable_precise_memory_
from tac_osm.compact_stream_state import compact_stream_state
from tac_osm.state_ingress import validate_restored_stream_state, StateIngressError
from research.state_ingress_20261110.reference import admissible


def model(seed=11,precise=False,epsilon=1e-8):
    torch.set_num_threads(1);torch.manual_seed(seed)
    cfg=CSL125R3Config(vocab_size=256,width=32,ff_width=64,layers=4,
        role_count=4,state_width=8,rungs=(2,4),attention_layers=(2,4),
        query_heads=4,kv_heads=2,attention_window=16,correction_rank=4,
        denominator_epsilon=epsilon)
    m=CSL125R4(cfg).eval()
    return enable_precise_memory_(m) if precise else m


def validate(m,s,t,depth=4):
    return validate_restored_stream_state(m,s,depth=depth,stream_ids=('owner',),observed_tokens=t)


def leaves(x):
    if isinstance(x,torch.Tensor):
        yield x
    elif is_dataclass(x):
        for f in fields(x):
            yield from leaves(getattr(x,f.name))
    elif isinstance(x,(tuple,list)):
        for v in x:
            yield from leaves(v)
    elif isinstance(x,dict):
        for v in x.values():
            yield from leaves(v)


@pytest.mark.parametrize('precise',[False,True])
@pytest.mark.parametrize('depth',[2,4])
@pytest.mark.parametrize('length',[0,17,65])
@pytest.mark.parametrize('seed',[11,22,33])
def test_actual_original_detached_compact_carry_and_continuation(seed,length,depth,precise):
    m=model(seed,precise);x=torch.randint(0,256,(1,length+1))
    with torch.no_grad():
        s=m.initial_state(1,torch.device('cpu'),torch.float32,depth=depth,stream_ids=('owner',))
        if length:
            _,s=m(x[:,:length],depth=depth,state=s,stream_ids=('owner',))
        rng=torch.get_rng_state().clone()
        ids=[id(t) for t in leaves(s)];values=[t.clone() for t in leaves(s)]
        for candidate in (s,s.detached(),compact_stream_state(s)):
            report=validate(m,candidate,length,depth)
            assert report['source_authentication']=='NOT_ESTABLISHED'
        assert ids==[id(t) for t in leaves(s)]
        assert all(torch.equal(a,b) for a,b in zip(values,leaves(s)))
        assert torch.equal(rng,torch.get_rng_state())
        a,sa=m(x[:,length:],depth=depth,state=s,stream_ids=('owner',),all_rungs=True,emit_repair_logits=True)
        validate(m,s,length,depth)
        b,sb=m(x[:,length:],depth=depth,state=s,stream_ids=('owner',),all_rungs=True,emit_repair_logits=True)
        assert all(torch.equal(u,v) for u,v in zip(leaves(a),leaves(b)))
        assert all(torch.equal(u,v) for u,v in zip(leaves(sa),leaves(sb)))


@pytest.mark.parametrize('fault,code', [('negative_mass','mass_negative'),('nan','memory_nonfinite'),
 ('infinity','memory_nonfinite'),('outside','numerator_outside_mass'),('zero_mass_nonzero_num','numerator_outside_mass'),
 ('dtype','memory_layout'),('meta','memory_layout'),('shape','memory_layout'),('conv_nan','conv_nonfinite'),
 ('KV_nan','KV_nonfinite'),('KV_cutoff','KV_cutoff'),('KV_bool','KV_cutoff'),('KV_length','KV_layout'),
 ('owner','stream_ownership'),('layers','state_layout'),('unknown','state_layout')])
def test_invalid_states_fail_without_repair(fault,code):
    m=model();x=torch.ones((1,7),dtype=torch.long)
    with torch.no_grad():
        _,s=m(x,depth=4,stream_ids=('owner',))
    s=deepcopy(s);layer=s.layers[0]
    if fault=='negative_mass':layer.fast_mass.fill_(-1)
    elif fault=='nan':layer.fast.fill_(float('nan'))
    elif fault=='infinity':layer.slow_mass.fill_(float('inf'))
    elif fault=='outside':layer.fast.copy_(layer.fast_mass+1)
    elif fault=='zero_mass_nonzero_num':layer.fast_mass.zero_();layer.fast.fill_(1)
    elif fault=='dtype':layer.fast=layer.fast.double()
    elif fault=='meta':layer.fast=torch.empty(layer.fast.shape,device='meta')
    elif fault=='shape':layer.fast=layer.fast[:,:,:1]
    elif fault=='conv_nan':layer.conv_tail.fill_(float('nan'))
    elif fault=='KV_nan':s.attention[1].k.fill_(float('nan'))
    elif fault=='KV_cutoff':s.attention[1].position=8
    elif fault=='KV_bool':s.attention[1].position=True
    elif fault=='KV_length':s.attention[1].k=s.attention[1].k[:,:,:1];s.attention[1].v=s.attention[1].v[:,:,:1]
    elif fault=='owner':s.stream_ids=('forged',)
    elif fault=='layers':s.layers=s.layers[:1]
    else:
        class Unknown(CSL125State):pass
        s=Unknown(s.layers,s.attention,s.stream_ids)
    with pytest.raises(StateIngressError,match='^'+code+'$'):
        validate(m,s,7)


@pytest.mark.parametrize('epsilon',[1e-100,1e-40])
def test_dtype_aware_normal_epsilon_contract(epsilon):
    for precise in (False,True):
        m=model(precise=precise,epsilon=epsilon)
        s=m.initial_state(1,torch.device('cpu'),torch.float32,depth=4,stream_ids=('owner',))
        if precise:validate(m,s,0)
        else:
            with pytest.raises(StateIngressError,match='epsilon_positive_normal'):validate(m,s,0)


@pytest.mark.parametrize('precise',[False,True])
def test_independent_numpy_cone_and_exact_boundary(precise):
    m=model(precise=precise);dtype=np.float64 if precise else np.float32
    s=m.initial_state(1,torch.device('cpu'),torch.float32,depth=4,stream_ids=('owner',))
    gen=np.random.default_rng(17)
    for case in range(20):
        d=gen.uniform(0,1,size=s.layers[0].fast.shape).astype(dtype)
        n=(d*gen.uniform(-1,1,size=d.shape)).astype(dtype)
        if case==0:d.fill(0);n.fill(-0.)
        elif case==1:n=d.copy()
        elif case%3==0:n.flat[0]=2
        elif case%5==0:d.flat[0]=-1
        s.layers[0].fast.copy_(torch.from_numpy(n));s.layers[0].fast_mass.copy_(torch.from_numpy(d))
        reference=bool(admissible(n,d,m.config.denominator_epsilon,dtype))
        try:validate(m,s,0);actual=True
        except StateIngressError:actual=False
        assert actual==reference


@pytest.mark.parametrize('seed',[11,22,33])
def test_validation_preserves_attached_parameter_and_incoming_state_gradients(seed):
    a=model(seed,True);b=deepcopy(a);x=torch.randint(0,256,(1,17))
    def run(m,guard):
        _,s=m(x[:,:5],depth=4,stream_ids=('owner',))
        incoming=list(leaves(s))
        for t in incoming:t.retain_grad()
        if guard:validate(m,s,5)
        out,_=m(x[:,5:],depth=4,state=s,stream_ids=('owner',),all_rungs=True,emit_repair_logits=True)
        sum(t.double().square().mean() for t in leaves(out)).backward()
        return [t.grad for t in incoming]
    ag=run(a,False);bg=run(b,True)
    for u,v in zip(ag,bg):
        assert u is not None and v is not None
        torch.testing.assert_close(u,v,atol=5e-6,rtol=5e-5)
    for (name,p),(name2,q) in zip(a.named_parameters(),b.named_parameters()):
        assert name==name2 and p.grad is not None and q.grad is not None
        torch.testing.assert_close(p.grad,q.grad,atol=5e-6,rtol=5e-5)


def test_cpu_BF16_autocast_existing_state_layout_and_invalid_context():
    m=model(precise=True)
    with torch.no_grad(),torch.autocast('cpu',dtype=torch.bfloat16):
        _,s=m(torch.ones((1,17),dtype=torch.long),depth=4,stream_ids=('owner',))
        validate(m,s,17)
    for key,value in [('depth',True),('observed_tokens',True),('stream_ids',('owner','owner'))]:
        args=dict(depth=4,observed_tokens=17,stream_ids=('owner',));args[key]=value
        with pytest.raises(StateIngressError):validate_restored_stream_state(m,s,**args)
