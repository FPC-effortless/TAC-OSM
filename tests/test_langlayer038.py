"""Constructed actual-graph controls; no trained prediction evidence."""
from pathlib import Path
import sys
import pytest
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'research/mathematical-engineering/langlayer038'))
from recurrence_only import RecurrenceOnlyDepth038, RecurrenceOnlyUpperBlock
from context_transition import ContextTransition026


def test_count_no_dummy_parameters_and_no_upper_concepts():
    m=RecurrenceOnlyDepth038();b=m.layernorm[1]
    assert m.parameter_count()==sum(p.numel() for p in m.parameters())==499901
    assert sum(p.numel() for p in b.parameters())==48041
    assert all(p.requires_grad for p in m.parameters())
    assert not any('concept' in n for n,_ in b.named_parameters())
    assert m.concept_q is not None and m.concept_k is not None
    assert m.ffn_up.out_features==442
    for other in (500122,499725,499661): assert abs(499901-other)/max(499901,other)<.01
    with pytest.raises(ValueError): RecurrenceOnlyDepth038(vocab=128)


@pytest.mark.parametrize('length',[1,7,17])
def test_prewrite_clock_and_serial_state(length):
    torch.manual_seed(38038);b=RecurrenceOnlyUpperBlock().double()
    z=torch.randn(2,length,280,dtype=torch.float64)
    _,tr=b(z,return_trace=True);state=torch.zeros(2,59,dtype=z.dtype)
    for t in range(length):
        assert torch.allclose(state,tr['prior_state'][:,t],atol=1e-12,rtol=1e-12)
        state=tr['alpha'][:,t]*state+(1-tr['alpha'][:,t])*tr['offer'][:,t]
        assert torch.allclose(state,tr['state'][:,t],atol=1e-12,rtol=1e-12)
    assert tr['state'].abs().max()<=1
    assert not tr['prior_state'][:,0].any()
    assert not any('concept' in n for n in tr)


def test_full_prediction_future_owner_reset_and_native_admission():
    torch.manual_seed(38038);m=RecurrenceOnlyDepth038().double().eval()
    x=torch.randint(0,256,(2,17));future=x.clone();future[:,9:]^=255
    other=x.clone();other[1]^=255
    mask=torch.ones_like(x,dtype=torch.bool);mask[:,4:7]=False
    with torch.no_grad():
        a,tr=m(x,write_mask=mask,return_trace=True)
        f=m(future,write_mask=mask);o=m(other,write_mask=mask)
        single=m(x[:1],write_mask=mask[:1]);again=m(x,write_mask=mask)
    assert torch.allclose(a[:,:9],f[:,:9],atol=1e-11,rtol=1e-9)
    assert torch.allclose(a[:1],o[:1],atol=1e-11,rtol=1e-9)
    assert torch.allclose(a[:1],single,atol=1e-11,rtol=1e-9)
    assert torch.equal(a,again)
    assert torch.equal(tr['native_memory'][:,3],tr['native_memory'][:,6])
    assert not tr['native_admission'][:,0].any()
    with pytest.raises(ValueError): m(x,start=4)


def test_live_gradients_and_past_influence():
    torch.manual_seed(38038);m=RecurrenceOnlyDepth038().double()
    x=torch.randint(0,256,(2,17));m(x).square().mean().backward()
    for name,p in m.layernorm[1].named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(),name
        assert p.grad.abs().max()>0,name
    assert m.context_lag_weights.grad.abs().max()>0 and m.ffn_out.weight.grad.abs().max()>0
    b=m.layernorm[1];z=torch.randn(2,17,280,dtype=torch.double);past=z.clone();past[:,0]+=torch.randn_like(past[:,0])
    with torch.no_grad(): a=b(z);p=b(past)
    assert not torch.allclose(a[:,1:3],p[:,1:3],atol=1e-12,rtol=0)


def test_inherited_original_core_unchanged():
    torch.manual_seed(38038);parent=ContextTransition026(ffn_width=442).double()
    m=RecurrenceOnlyDepth038().double();weights=m.state_dict()
    for name,value in parent.state_dict().items():
        destination='layernorm.0.'+name.split('.',1)[1] if name.startswith('layernorm.') else name
        weights[destination]=value
    m.load_state_dict(weights)
    x=torch.randint(0,256,(2,7))
    with torch.no_grad(): _,a=parent(x,return_trace=True);_,b=m(x,return_trace=True)
    assert set(a)==set(b)
    for name in a: assert torch.equal(a[name],b[name]),name


def test_reject_invalid_upper_inputs():
    b=RecurrenceOnlyUpperBlock()
    for x in (torch.zeros(2,0,280),torch.zeros(2,4,59),torch.zeros(2,4,280,dtype=torch.long)):
        with pytest.raises(ValueError): b(x)
    with pytest.raises(ValueError): RecurrenceOnlyUpperBlock(state_width=0)


def test_actual_upper_probe_matches_source_scan_and_is_pure():
    from probe_adapter038 import UPPER_PROBE
    from tac_osm.research_probe import probe_memory
    torch.manual_seed(38038);b=RecurrenceOnlyUpperBlock().double().eval()
    z=torch.randn(2,7,280,dtype=torch.float64)
    sample={'observed_features':z[:,:-1],'query_features':z[:,-1]}
    UPPER_PROBE.validate(b,sample);h=UPPER_PROBE.start(b,sample)
    with torch.no_grad():
        for t in range(6): h=UPPER_PROBE.advance(b,h,sample,t)
        out=UPPER_PROBE.forecast(b,h,sample)['residual_features']
        full=b(z)
    assert torch.allclose(full[:,-1],out,atol=1e-11,rtol=1e-9)
    report=probe_memory(b,sample,UPPER_PROBE)
    assert report['query_does_not_write_state']
    assert report['candidate_permutation']['status']=='INCONCLUSIVE_NOT_DECLARED'
    assert report['state_components']==['upper_recurrence']
