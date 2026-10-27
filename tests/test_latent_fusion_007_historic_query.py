"""LATFUSE007 mathematical and source-boundary tests; no synthetic skill."""
import pytest
import torch
import torch.nn.functional as F
from tac_osm.latent_fusion_004_factorial import FactorialFusion
from tac_osm.latent_fusion_007_historic_query import SourceTimedEventQuery

def test_original_fp32_recurrent_memory_unchanged_and_exact_live_parameters():
    c=SourceTimedEventQuery().original_state_and_weight_contract()
    assert c["original_recurrent_fp32_values"]==1696
    assert c["original_bool_source_flags"]==16
    assert c["original_nominal_memory_bytes"]==6800
    assert c["trainable_parameters"]==c["deployed_parameters"]==99390
    assert c["original_beta_and_alpha_writer_unchanged"]

def test_inherited_original_writer_and_future_forecast_bitwise_identical():
    torch.manual_seed(73);a=FactorialFusion("2h-soft")
    torch.manual_seed(73);b=SourceTimedEventQuery()
    for name,value in a.state_dict().items():
        assert torch.equal(value,b.state_dict()[name]),name
    orig=a.initial(2);new=b.initial(2)
    for _ in range(3):
        sensor=torch.randn(2,13)
        mask=torch.ones_like(sensor,dtype=torch.bool)
        zeros=torch.zeros_like(sensor)
        src=torch.tensor([0,1])
        orig=a.step(orig,sensor,mask,zeros,zeros,src)
        new=b.step(new,sensor,mask,zeros,zeros,src)
        for x,y in zip(orig,new):torch.testing.assert_close(x,y,atol=0,rtol=0)
    q=torch.randn(2,3,6)
    for x,y in zip(a.forecast(orig,src,q),b.forecast(new,src,q)):
        torch.testing.assert_close(x,y,atol=0,rtol=0)

def test_historic_queries_permute_without_mutation():
    torch.manual_seed(17)
    m=SourceTimedEventQuery()
    s=m.initial(3)
    for _ in range(4):
        x=torch.randn(3,13);seen=torch.ones_like(x,dtype=torch.bool)
        zero=torch.zeros_like(x)
        s=m.step(s,x,seen,zero,zero,torch.tensor([0,1,2]))
    old=[t.detach().clone() for t in s]
    cal=torch.randn(3,4,6)
    out=m.historic_event_logits(s,torch.tensor([0,1,2]),cal)
    out2=m.historic_event_logits(s,torch.tensor([0,1,2]),cal[:,[3,1,0,2]])
    torch.testing.assert_close(out[:,[3,1,0,2]],out2,atol=5e-7,rtol=1e-5)
    assert all(torch.equal(v,x) for v,x in zip(old,s))

def test_no_associative_memory_means_no_past_clock_content_signal():
    m=SourceTimedEventQuery()
    s=m.initial(3)
    a=torch.randn(3,2,6)
    result=m.historic_event_logits(s,torch.tensor([0,1,2]),a)
    # With exactly zero matrix, time-specific differences cannot reach
    # content-only historic head; it returns its learned bias.
    assert torch.equal(result,result[:,0:1].expand_as(result))

def test_registered_historic_readout_has_gradient_to_original_2head_memory():
    torch.manual_seed(7);m=SourceTimedEventQuery()
    s=m.initial(3)
    for j in range(6):
        x=torch.randn(3,13)+j*.4
        obs=torch.ones_like(x,dtype=torch.bool)
        s=m.step(s,x,obs,torch.zeros_like(x),torch.zeros_like(x),torch.arange(3))
    logits=m.historic_event_logits(s,torch.arange(3),torch.randn(3,1,6))
    loss=F.binary_cross_entropy_with_logits(logits[:,0],torch.tensor([1.,0.,1.]))
    loss.backward()
    for name in ("key_map","val_map","read_router","historic_clock_query","historic_event_head"):
        for p in getattr(m,name).parameters():
            assert p.grad is not None and torch.isfinite(p.grad).all(),name

def test_invalid_source_id_historic_clock_and_dtype_fail_closed():
    m=SourceTimedEventQuery();s=m.initial(2)
    with pytest.raises(ValueError):
        m.historic_event_logits(s,torch.tensor([0,16]),torch.zeros(2,1,6))
    with pytest.raises(ValueError):
        m.historic_event_logits(s,torch.tensor([0,1]),torch.zeros(2,1,5))
    with pytest.raises(ValueError):
        m.historic_event_logits(s,torch.tensor([0,1]),torch.zeros(2,1,6,dtype=torch.float64))
