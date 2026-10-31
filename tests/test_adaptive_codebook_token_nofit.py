"""SENSE-TOK-005 committed no-fit regression subset; no optimizer used."""
import pytest
import torch
from torch.nn import functional as F
from tac_osm.adaptive_byte_token import AdaptiveByteTokenGRU
from tac_osm.adaptive_codebook_token import AdaptiveCodebookByteTokenGRU,codebook_parameter_receipt
from tac_osm.causal_byte_probe import predecision_examples
from tac_osm.adaptive_codebook_math import position_pool_reference,code_id_reference


def inputs(raw):
    x,_,_=predecision_examples(raw,'fixed')
    return x[None]


def test_01_parameter_receipt_untrained():
    m=AdaptiveCodebookByteTokenGRU(code_count=8)
    r=codebook_parameter_receipt(m)
    assert r['trainable_parameters']==101008
    assert r['deployed_learned_parameters']==101008
    assert r['within_100k_band'] and r['within_1pct_s004']
    assert r['training_authorized'] is False


@pytest.mark.parametrize('force',('byte','token',None))
def test_02_byte_token_routing_causal_and_lossless(force):
    torch.manual_seed(21)
    m=AdaptiveCodebookByteTokenGRU(max_token_bytes=4).eval()
    raw=b'abc123!?abc123!?'
    with torch.no_grad():
        o,(a,),_=m(inputs(raw),force=force,audit=True)
    a.verify(raw[:-1])
    assert o.shape==(1,len(raw),256)
    assert torch.isfinite(o).all()
    assert a.base.byte_readouts==len(raw)
    assert all(0<=c.end<=c.available_after<=len(raw)-1 for c in a.codes)
    if force=='byte': assert not a.codes
    if force=='token': assert len(a.codes)>=1


def test_03_surrogate_forward_equals_discrete_inference():
    torch.manual_seed(22)
    m=AdaptiveCodebookByteTokenGRU(max_token_bytes=4).eval()
    x=inputs(b'abc123xyz456abc123')
    with torch.no_grad():
        a,aa,_=m(x,mode='inference',audit=True)
        b,bb,_=m(x,mode='surrogate',audit=True)
    assert torch.allclose(a,b,atol=1e-6,rtol=1e-6)
    assert aa[0].codes==bb[0].codes


def test_04_unseen_future_cannot_change_logits_or_prior_codes():
    torch.manual_seed(23)
    m=AdaptiveCodebookByteTokenGRU(max_token_bytes=4).eval()
    p=b'abc123xyz456'
    with torch.no_grad():
        a,(ra,),_=m(inputs(p+b'AAAAAAAA'),audit=True)
        b,(rb,),_=m(inputs(p+b'ZZZZZZZZ'),audit=True)
    assert torch.equal(a[:,:len(p)+1],b[:,:len(p)+1])
    assert ra.base.decisions[:len(p)]==rb.base.decisions[:len(p)]
    assert [c for c in ra.codes if c.available_after<=len(p)]==[c for c in rb.codes if c.available_after<=len(p)]


def test_05_codebook_knockout_affects_token_not_byte_path():
    torch.manual_seed(24)
    m=AdaptiveCodebookByteTokenGRU(max_token_bytes=3).eval()
    x=inputs(b'aaabbbcccddd')
    with torch.no_grad():
        t0=m(x,force='token')
        b0=m(x,force='byte')
        m.token_codes.weight.zero_()
        t1=m(x,force='token')
        b1=m(x,force='byte')
    assert not torch.equal(t0,t1)
    assert torch.equal(b0,b1)


def test_06_active_task_gradient_enters_codebook():
    torch.manual_seed(25)
    m=AdaptiveCodebookByteTokenGRU(max_token_bytes=3)
    raw=b'abc123xyz0123456789\n'*2
    x,_,y=predecision_examples(raw,'fixed')
    out=m(x[None],mode='surrogate',force='token')
    F.cross_entropy(out[0],y).backward()
    assert m.token_codes.weight.grad is not None
    assert m.token_codes.weight.grad.abs().sum().item()>0


def test_07_active_task_gradient_enters_controller():
    torch.manual_seed(26)
    m=AdaptiveCodebookByteTokenGRU(max_token_bytes=3)
    with torch.no_grad(): m.mode_gate.bias.fill_(-1)
    raw=b'abc123xyz0123456789\n'*2
    x,_,y=predecision_examples(raw,'fixed')
    out=m(x[None],mode='surrogate')
    F.cross_entropy(out[0],y).backward()
    assert m.mode_gate.weight.grad is not None
    assert m.mode_gate.weight.grad.abs().sum().item()>0


def test_08_counterfactual_decisions_have_identical_prior_predictions():
    torch.manual_seed(27)
    m=AdaptiveCodebookByteTokenGRU(max_token_bytes=8).eval()
    with torch.no_grad(): m.mode_gate.bias.fill_(-2)
    x=inputs(b'abc123def456xyz')
    t=4
    with torch.no_grad():
        a,(ra,),_=m(x,intervention_at=t,intervention_mode='byte',audit=True)
        b,(rb,),_=m(x,intervention_at=t,intervention_mode='token',audit=True)
    assert torch.equal(a[:,:t],b[:,:t])
    assert ra.base.decisions[t-1]=='BYTE' and rb.base.decisions[t-1]=='TOKEN'
    assert not torch.equal(a[:,t:],b[:,t:])


def test_09_reference_matches_actual_codebook_assignment():
    torch.manual_seed(28)
    m=AdaptiveCodebookByteTokenGRU(code_count=6).eval()
    with torch.no_grad():
        m.position_recency.uniform_(-.6,.6)
        xs=m.byte_emb(torch.tensor([97,98,99]))
        items=[row.unsqueeze(0) for row in xs.unbind(0)]
        expected=position_pool_reference([x[0].tolist() for x in items],m.position_recency.tolist())
        actual=m._pool(items)[0].tolist()
        assert max(abs(a-b) for a,b in zip(actual,expected))<1e-6
        _,idx=m._assignment(items)
        assert idx==code_id_reference(expected,m.token_codes.weight.tolist())


def test_10_reject_future_oracle_action_and_nonbyte_inputs():
    m=AdaptiveCodebookByteTokenGRU()
    x=inputs(b'abc123')
    with pytest.raises(ValueError):
        m(x,intervention_at=2,intervention_mode='gold')
    with pytest.raises(ValueError):
        m(x,force='byte',intervention_at=2,intervention_mode='token')
    with pytest.raises(ValueError):
        m(torch.tensor([[65,66]]))
