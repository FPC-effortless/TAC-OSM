"""LATFUSE008 paired model contracts; constructed unit inputs are NOT performance claims."""
import torch
import pytest
from tac_osm.latent_fusion_007_historic_query import SourceTimedEventQuery
from tac_osm.latent_fusion_008_nonassoc_control import SourceTimedNonassociativeQuery

def pair(seed=808):
    torch.manual_seed(seed)
    original=SourceTimedEventQuery()
    torch.manual_seed(seed)
    baseline=SourceTimedNonassociativeQuery()
    return original,baseline

def test_identical_live_parameter_and_original_FP32_state_contract():
    for m in pair():
        c=m.original_state_and_weight_contract()
        assert c["original_recurrent_fp32_values"]==1696
        assert c["original_bool_source_flags"]==16
        assert c["original_nominal_memory_bytes"]==6800
        assert c["trainable_parameters"]==c["deployed_parameters"]==99390

def test_identical_initial_weights_and_original_source_writer_and_future_forecast():
    a,b=pair()
    for k,v in a.state_dict().items():
        assert torch.equal(v,b.state_dict()[k]),k
    states=[a.initial(2),b.initial(2)]
    for _ in range(3):
        x=torch.randn(2,13)
        mask=torch.ones_like(x,dtype=torch.bool)
        t=torch.zeros_like(x)
        for i,m in enumerate((a,b)):
            states[i]=m.step(states[i],x,mask,t,t,torch.tensor([0,1]))
        for p,q in zip(*states):
            assert torch.equal(p,q)
    future=torch.randn(2,1,6)
    for p,q in zip(a.forecast(states[0],torch.tensor([0,1]),future),
                   b.forecast(states[1],torch.tensor([0,1]),future)):
        assert torch.equal(p,q)

def test_history_event_control_does_not_read_associative_storage():
    _,b=pair()
    s=b.initial(2)
    for i in range(4):
        v=torch.randn(2,13)+i
        m=torch.ones_like(v,dtype=torch.bool)
        s=b.step(s,v,m,torch.zeros_like(v),torch.zeros_like(v),torch.tensor([0,1]))
    changed=s._replace(associative=torch.randn_like(s.associative)*3)
    clock=torch.randn(2,3,6)
    y=b.historic_event_logits(s,torch.tensor([0,1]),clock)
    y2=b.historic_event_logits(changed,torch.tensor([0,1]),clock)
    assert torch.equal(y,y2)

def test_memory_variant_actual_historic_read_changes_with_matrix():
    a,_=pair()
    s=a.initial(2)
    for i in range(4):
        v=torch.randn(2,13)+i
        m=torch.ones_like(v,dtype=torch.bool)
        s=a.step(s,v,m,torch.zeros_like(v),torch.zeros_like(v),torch.tensor([0,1]))
    blank=s._replace(associative=torch.zeros_like(s.associative))
    clock=torch.randn(2,3,6)
    actual=a.historic_event_logits(s,torch.tensor([0,1]),clock)
    zero=a.historic_event_logits(blank,torch.tensor([0,1]),clock)
    assert torch.isfinite(actual).all() and torch.isfinite(zero).all()
    assert not torch.equal(actual,zero)

def test_noassoc_historic_read_uses_current_fast_and_slow():
    _,b=pair()
    s=b.initial(2)
    def query(s):return b.historic_event_logits(s,torch.tensor([0,1]),torch.randn(2,1,6))
    clock=torch.randn(2,1,6)
    a=b.historic_event_logits(s._replace(fast=torch.ones_like(s.fast),slow=torch.ones_like(s.slow)),
                              torch.tensor([0,1]),clock)
    c=b.historic_event_logits(s._replace(fast=torch.zeros_like(s.fast),slow=torch.zeros_like(s.slow)),
                              torch.tensor([0,1]),clock)
    assert not torch.equal(a,c)

def test_matched_baseline_historic_parameters_receive_gradients():
    _,b=pair()
    state=b.initial(2)
    for i in range(4):
        v=torch.randn(2,13)+i
        state=b.step(state,v,torch.ones_like(v,dtype=torch.bool),
                     torch.zeros_like(v),torch.zeros_like(v),torch.tensor([0,1]))
    logits=b.historic_event_logits(state,torch.tensor([0,1]),torch.randn(2,1,6))
    loss=torch.nn.functional.binary_cross_entropy_with_logits(logits[:,0],torch.tensor([0.,1.]))
    loss.backward()
    for name in ("historic_clock_query","historic_event_head","belief_query","read_router","temporal_a","slow_input"):
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in getattr(b,name).parameters()),name

def test_invalid_historic_clock_fails_closed():
    _,b=pair()
    with pytest.raises(ValueError):
        b.historic_event_logits(b.initial(2),torch.tensor([0,1]),torch.zeros(2,1,6,dtype=torch.float64))
