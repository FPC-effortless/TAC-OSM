import json
import math

import pytest
import torch

from scripts.lacuna_retrieval003 import (ROOT, RawRing, attention, magnitude_match,
    override_slow_read, patched_logits, reconstruct)
from scripts.replay_langmem086087 import load_source


def model():
    torch.manual_seed(639)
    manifest = json.loads((ROOT/'research/replication/langmem086087/MANIFEST.json').read_text())
    return load_source(ROOT, manifest)().eval()


def test_wrapped_ring_charges_clock_and_requires_complete_past_context():
    ring = RawRing()
    for t in range(900):
        ring.append(t % 256)
    assert len(ring.data) == 256
    assert ring.clock == 900
    assert ring.candidates()[0] == 656
    assert ring.candidates()[-1] == 898  # current write 899 excluded
    assert ring.context(656) == bytes(i % 256 for i in range(652, 656))
    with pytest.raises(IndexError):
        ring.context(655)


@torch.no_grad()
def test_reconstructed_keys_and_values_match_real_source_without_state_cache():
    m = model()
    raw = b'Prefix bytes with repeated actual context. ' * 8
    ring = RawRing()
    state = m.start(('fixture',))
    keys, values = {}, {}
    for t, byte in enumerate(raw):
        _, state, trace = m(torch.tensor([[byte]]), state=state, owners=('fixture',), return_trace=True)
        ring.append(byte)
        keys[t], values[t] = trace[0]['writer'][0], trace[0]['value'][0]
    ids = ring.candidates()
    k, v = reconstruct(m, ring, ids)
    assert torch.allclose(k.float(), torch.stack([keys[j] for j in ids]), atol=3e-6, rtol=3e-6)
    assert torch.allclose(v.float(), torch.stack([values[j] for j in ids]), atol=3e-6, rtol=3e-6)


@torch.no_grad()
def test_patch_covers_source_consumers_identity_zero_and_removes_hooks_on_error():
    m = model()
    _, prior = m(torch.tensor([[97,98,99,100]]), owners=('fixture',))
    token = torch.tensor([[101]])
    logits, _, trace = m(token, state=prior, owners=('fixture',), return_trace=True)
    equal = patched_logits(m, token, prior, 'fixture', trace[0], trace[0]['read_slow'][0])
    zero = patched_logits(m, token, prior, 'fixture', trace[0], torch.zeros(8))
    source_zero = m(token, state=prior, owners=('fixture',), lesion='slow_zero')[0][0,0]
    assert torch.equal(logits[0,0], equal)
    assert torch.allclose(zero, source_zero, atol=3e-6, rtol=3e-6)
    with pytest.raises(RuntimeError):
        with override_slow_read(m, trace[0], torch.zeros(8)):
            raise RuntimeError('abort fixture')
    assert all(not getattr(m,name)._forward_pre_hooks for name in ('fast','slow','norm'))


def test_attention_and_norm_matching_against_independent_reference():
    from scripts.lacuna_retrieval003 import REG
    import importlib.util
    spec = importlib.util.spec_from_file_location('independent003', REG/'pnds_reference.py')
    ref = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ref)
    generator = torch.Generator().manual_seed(639)
    for _ in range(30):
        keys = torch.nn.functional.normalize(torch.randn(25,8,generator=generator,dtype=torch.float64),dim=1)
        values = torch.randn(25,8,generator=generator,dtype=torch.float64)
        query = torch.nn.functional.normalize(torch.randn(8,generator=generator,dtype=torch.float64),dim=0)
        vector, weights = attention(keys, values, query)
        expected, ew = ref.attention(keys.tolist(), values.tolist(), query.tolist())
        assert max(abs(a-b) for a,b in zip(vector.tolist(),expected)) < 1e-12
        assert max(abs(a-b) for a,b in zip(weights.tolist(),ew)) < 1e-12
        reference = query*.037
        matched = magnitude_match(vector, reference)
        assert math.isclose(float(matched.norm()),float(reference.norm()),abs_tol=1e-12)
        assert max(abs(a-b) for a,b in zip(matched.tolist(),ref.match_norm(vector.tolist(),reference.tolist()))) < 1e-12
    assert torch.equal(magnitude_match(torch.zeros(8),torch.ones(8)),torch.zeros(8))


def test_identical_addresses_cannot_identify_distinct_values_by_this_read_rule():
    key = torch.nn.functional.normalize(torch.ones(8, dtype=torch.float64),dim=0)
    values = torch.eye(8,dtype=torch.float64)[:2]
    read, weight = attention(torch.stack((key,key)),values,key)
    assert torch.equal(weight,torch.tensor([.5,.5],dtype=torch.float64))
    assert torch.equal(read,values.mean(0))
    assert torch.equal(attention(torch.stack((key,key)),values.roll(1,0),key)[0],read)
