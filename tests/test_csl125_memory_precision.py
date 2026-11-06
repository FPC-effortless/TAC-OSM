"""Targeted R4 precision-engineering checks, not learned-capability evaluation."""
from copy import deepcopy
from unittest.mock import patch
import pytest
import torch
from tac_osm.csl125_r3 import CSL125R3Config
from tac_osm.csl125_tool_model import CSL125R4
from tac_osm.csl125_memory_precision import (
    PreciseMemoryState, PreciseRoleStateBlock, enable_precise_memory_,
    precise_affine_scan, precise_normalized_read,
)
from scripts.audit_csl_scan_cancellation import serial_torch


def cfg():
    return CSL125R3Config(vocab_size=256, width=32, ff_width=64, layers=4,
        role_count=4, state_width=8, rungs=(2,4), attention_layers=(2,4),
        query_heads=4, kv_heads=2, attention_window=16, correction_rank=4)


def serial_memory(z, carry, rate, gate, chunk):
    return serial_torch(z.double(), carry.double(), rate.double(), gate.double(), chunk)


def test_installation_preserves_all_parameter_objects_values_and_names():
    torch.manual_seed(61)
    model = CSL125R4(cfg()).eval()
    before = {n: id(p) for n,p in model.named_parameters()}
    weights = {n: t.clone() for n,t in model.state_dict().items()}
    rng = torch.get_rng_state().clone()
    assert enable_precise_memory_(model) is model
    assert {n: id(p) for n,p in model.named_parameters()} == before
    assert tuple(weights) == tuple(model.state_dict())
    assert all(torch.equal(t, model.state_dict()[n]) for n,t in weights.items())
    assert torch.equal(rng, torch.get_rng_state())
    assert all(p.dtype == torch.float32 for p in model.parameters())
    assert all(type(b) is PreciseRoleStateBlock for b in model.blocks)
    enable_precise_memory_(model)
    assert {n: id(p) for n,p in model.named_parameters()} == before


def test_default_full_size_meta_count_unchanged():
    model = CSL125R4(device='meta')
    assert model.parameter_count() == 123_873_263
    enable_precise_memory_(model)
    assert model.parameter_count() == 123_873_263
    assert model.parameter_count() <= 125_000_000


def test_old_carry_rejected_and_new_dtype_retained_through_detach():
    model = CSL125R4(cfg()).eval()
    tokens = torch.randint(0,256,(1,7))
    with torch.no_grad():
        _, old = model(tokens, depth=4, stream_ids=('one',))
        enable_precise_memory_(model)
        with pytest.raises(ValueError, match='reset/replay'):
            model(tokens, depth=4, state=old, stream_ids=('one',))
        _, state = model(tokens, depth=4, stream_ids=('one',))
    detached = state.detached()
    for s in detached.layers:
        assert isinstance(s, PreciseMemoryState)
        assert all(getattr(s,k).dtype == torch.float64
                   for k in ('fast','slow','fast_mass','slow_mass'))
        assert s.conv_tail.dtype == torch.float32
    with pytest.raises(ValueError, match='stream'):
        model(tokens, depth=4, state=detached, stream_ids=('other',))


@pytest.mark.parametrize('seed',[61,62,63])
def test_all_participating_model_parameter_gradients_match_independent_serial(seed):
    torch.manual_seed(seed)
    model = enable_precise_memory_(CSL125R4(cfg())).train()
    reference = deepcopy(model)
    tokens = torch.randint(0,256,(2,17))
    mask = torch.zeros((2,16), dtype=torch.bool); mask[:,8:] = True
    loss = model.tool_sft_loss(tokens, mask, correction_weight=0.25)
    loss.backward()
    with patch('tac_osm.csl125_memory_precision.precise_affine_scan', serial_memory):
        expected = reference.tool_sft_loss(tokens, mask, correction_weight=0.25)
        expected.backward()
    assert torch.allclose(loss, expected, atol=5e-6, rtol=5e-5)
    compared = 0
    for (name,p), (n,q) in zip(model.named_parameters(), reference.named_parameters()):
        assert name == n
        assert (p.grad is None) == (q.grad is None), name
        if p.grad is not None:
            assert torch.isfinite(p.grad).all(), name
            assert torch.allclose(p.grad, q.grad, atol=5e-6, rtol=5e-5), name
            compared += 1
    assert compared > 50


def test_future_poison_and_chunked_stream_agreement():
    torch.manual_seed(64)
    model = enable_precise_memory_(CSL125R4(cfg())).eval()
    x = torch.randint(0,256,(1,31)); poison=x.clone();poison[:,20:] = (poison[:,20:]+17)%256
    with torch.no_grad():
        full, final = model(x, all_rungs=True, stream_ids=('one',))
        y,_ = model(poison, all_rungs=True, stream_ids=('one',))
        for rung in model.config.rungs:
            assert torch.equal(full[rung]['token_logits'][:,:20], y[rung]['token_logits'][:,:20])
        state=None; parts=[]
        for lo,hi in ((0,7),(7,20),(20,31)):
            o,state = model(x[:,lo:hi],depth=4,state=state,stream_ids=('one',))
            parts.append(o[4]['token_logits'])
        assert torch.allclose(torch.cat(parts,1), full[4]['token_logits'],atol=3e-4,rtol=3e-5)
        for a,b in zip(state.layers, final.layers):
            for key in ('fast','slow','fast_mass','slow_mass'):
                assert torch.allclose(getattr(a,key),getattr(b,key),atol=3e-4,rtol=3e-5)


def test_cpu_bf16_autocast_preserves_memory_boundary_and_finite_gradient():
    torch.manual_seed(65)
    model=enable_precise_memory_(CSL125R4(cfg())).train()
    tokens=torch.randint(0,256,(2,24))
    with torch.autocast('cpu',dtype=torch.bfloat16):
        outputs,state=model(tokens,all_rungs=True,emit_repair_logits=True)
        loss=sum(o['token_logits'].float().square().mean() +
                 o['repair_logits'].float().square().mean() +
                 o['tool_logits'].float().square().mean() +
                 o['field_logits'].float().square().mean() +
                 o['accept_logit'].float().square().mean() +
                 o['error_logit'].float().square().mean() for o in outputs.values())
    assert torch.isfinite(loss)
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    assert all(p.dtype == torch.float32 for p in model.parameters())
    assert all(s.fast.dtype == s.fast_mass.dtype == torch.float64 for s in state.layers)


def test_closed_gate_holds_state_and_invalid_scan_fails_closed():
    z=torch.randn(1,5,2,3,2);carry=torch.randn(1,2,3,2,dtype=torch.float64)
    rate=torch.ones(2,3)*.1;gate=torch.zeros(1,5,2)
    out,end=precise_affine_scan(z,carry,rate,gate,3)
    assert torch.equal(end,carry)
    assert torch.equal(out,carry[:,None].expand_as(out))
    with pytest.raises(ValueError):
        precise_affine_scan(z,carry,rate,gate,65)
    with pytest.raises(ValueError):
        precise_affine_scan(z,carry,rate,gate-1,3)
    with pytest.raises(ValueError):
        precise_normalized_read(out.float(),torch.ones(1,5,2)/2,1e-8)
