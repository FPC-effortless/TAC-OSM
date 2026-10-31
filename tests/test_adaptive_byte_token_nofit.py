"""SENSE-TOK-004: no-fit falsifiers; deliberately never train any weights."""
from __future__ import annotations
import random
import pytest
import torch
import torch.nn.functional as F

from tac_osm.causal_byte_probe import predecision_examples
from tac_osm.adaptive_byte_token import AdaptiveByteTokenGRU, adaptive_parameter_receipt


def make_ids(raw):
    return predecision_examples(raw, 'fixed')[0][None]


def test_00_parameter_budget_actual_active_model():
    m = AdaptiveByteTokenGRU()
    r = adaptive_parameter_receipt(m)
    assert r['trainable_parameters'] == r['deployed_learned_parameters'] == 100752
    assert r['within_100k_band'] and r['within_1pct_vs_sparse']
    assert not r['training_authorized'] and not r['semantic_tokenization_demonstrated']


@pytest.mark.parametrize('mode', ['inference', 'surrogate'])
@pytest.mark.parametrize('forced', ['byte', 'token', None])
def test_01_all_modes_all_decisions_lossless(mode, forced):
    raw = bytes(range(1, 41))
    m = AdaptiveByteTokenGRU(max_token_bytes=4)
    x = make_ids(raw)
    logits, (a,), cost = m(x, mode=mode, force=forced, audit=True)
    assert logits.shape == (1, len(raw), 256) and torch.isfinite(logits).all()
    assert torch.isfinite(cost)
    assert a.observed_bytes == len(raw) - 1
    assert a.byte_readouts == len(raw)
    assert a.recurrent_updates == len(a.events)
    a.verify(raw[:-1])
    assert b''.join(e.payload for e in a.events) + a.pending == raw[:-1]
    assert all(ev.available_after >= ev.end for ev in a.events)
    assert a.eos_used is False
    if forced == 'byte':
        assert all(ev.mode == 'BYTE' and len(ev.payload) == 1 for ev in a.events)
        assert a.recurrent_updates == len(raw) - 1
    if forced == 'token':
        assert all(ev.forced and len(ev.payload) == 4 for ev in a.events)
        assert len(a.pending) == (len(raw) - 1) % 4
        assert a.recurrent_updates == (len(raw) - 1) // 4


def test_02_unseen_next_byte_poison_does_not_change_prediction_or_actions():
    torch.manual_seed(102)
    model = AdaptiveByteTokenGRU().eval()
    observed = b'abc123_hello_'
    a = make_ids(observed + b'unknown1??')
    b = make_ids(observed + b'different!')
    with torch.no_grad():
        la, (aa,), _ = model(a, audit=True)
        lb, (ab,), _ = model(b, audit=True)
    assert torch.equal(la[0, :len(observed)+1], lb[0, :len(observed)+1])
    assert aa.decisions[:len(observed)] == ab.decisions[:len(observed)]
    assert aa.byte_probabilities[:len(observed)] == ab.byte_probabilities[:len(observed)]


def test_03_prefix_replay_match_for_inference():
    torch.manual_seed(7)
    model = AdaptiveByteTokenGRU().eval()
    x = make_ids(b'abc123-\n' * 5)
    with torch.no_grad():
        all_logits = model(x)
        for n in (1, 2, 3, 4, 7, 15, 30, x.shape[1]):
            part = model(x[:, :n])
            assert torch.equal(part, all_logits[:, :n]), n


@pytest.mark.parametrize('force', ['byte', 'token', None])
def test_04_surrogate_forward_is_exactly_inference_choice(force):
    torch.manual_seed(91)
    model = AdaptiveByteTokenGRU(max_token_bytes=5).eval()
    x = make_ids(b'aaaaaaaaab123456789_?\n' * 2)
    with torch.no_grad():
        li, (ai,), _ = model(x, mode='inference', force=force, audit=True)
        ls, (ass,), _ = model(x, mode='surrogate', force=force, audit=True)
    assert torch.allclose(li, ls, atol=1e-6, rtol=1e-6)
    assert ai.events == ass.events
    assert ai.decisions == ass.decisions
    assert ass.candidate_recurrent_calls >= ai.candidate_recurrent_calls


def test_05_gate_receives_gradient_from_byte_prediction_without_cost_penalty():
    torch.manual_seed(101)
    m = AdaptiveByteTokenGRU(max_token_bytes=6).train()
    raw = b'abc123def456++--\n' * 3
    ids, _, targets = predecision_examples(raw, 'fixed')
    logits, _, _ = m(ids[None], mode='surrogate', audit=True)
    loss = F.cross_entropy(logits[0], targets)
    assert torch.isfinite(loss)
    loss.backward()
    for name, param in m.named_parameters():
        assert param.grad is not None and torch.isfinite(param.grad).all(), name
        assert param.grad.abs().sum().item() > 0, name
    assert m.mode_gate.weight.grad.abs().sum().item() > 0


def test_06_gate_receives_gradient_from_cost_proxy():
    torch.manual_seed(101)
    m = AdaptiveByteTokenGRU()
    x = make_ids(b'abcdefgh1234567')
    _, _, proxy = m(x, mode='surrogate', audit=True)
    proxy.backward()
    assert m.mode_gate.weight.grad is not None
    assert m.mode_gate.weight.grad.abs().sum().item() > 0


def test_07_controller_switch_actually_changes_output():
    torch.manual_seed(21)
    m = AdaptiveByteTokenGRU().eval()
    x = make_ids(b'abcdefgh12345678@@' * 2)
    with torch.no_grad():
        byte, (ab,), _ = m(x, force='byte', audit=True)
        token, (at,), _ = m(x, force='token', audit=True)
    assert ab.recurrent_updates > at.recurrent_updates
    assert not torch.equal(byte, token)
    assert ab.events != at.events


def test_08_gate_is_observed_context_dependent_even_without_fit():
    # Structured feature-routing source check, NOT evidence of trained skill.
    torch.manual_seed(21)
    m = AdaptiveByteTokenGRU().eval()
    xa = make_ids(b'a' * 23)
    xb = make_ids(b'z' * 23)
    with torch.no_grad():
        _, (a,), _ = m(xa, audit=True)
        _, (b,), _ = m(xb, audit=True)
    assert a.byte_probabilities != b.byte_probabilities


def test_09_batch_isolation_and_no_hidden_shared_history():
    torch.manual_seed(41)
    m = AdaptiveByteTokenGRU().eval()
    a = make_ids(b'abc123456_abc123456')
    b = make_ids(b'zzz4444555zzz444455')
    with torch.no_grad():
        batch = m(torch.cat([a, b], dim=0))
        single = m(a)
    assert torch.equal(batch[0], single[0])


def test_10_invalid_inputs_no_silent_masks_or_eos():
    m = AdaptiveByteTokenGRU()
    with pytest.raises(ValueError):
        m(torch.tensor([[97, 98]]))
    with pytest.raises(ValueError):
        m(torch.tensor([[256, 256]]))
    with pytest.raises(ValueError):
        m(make_ids(b'abc'), mode='oracle')
    with pytest.raises(ValueError):
        m(make_ids(b'abc'), force='future_label')
    with pytest.raises(ValueError):
        AdaptiveByteTokenGRU(max_token_bytes=1)
    with pytest.raises(ValueError):
        AdaptiveByteTokenGRU(threshold=1.0)
    with pytest.raises(ValueError):
        AdaptiveByteTokenGRU(temperature=float('nan'))
    assert m(torch.empty((0, 0), dtype=torch.long)).shape == (0, 0, 256)
    assert m(torch.empty((1, 0), dtype=torch.long)).shape == (1, 0, 256)


def test_11_randomized_sequences_and_finite_logits():
    rng = random.Random(20261031)
    m = AdaptiveByteTokenGRU(max_token_bytes=7)
    for n in (1, 2, 3, 7, 8, 15, 17, 40, 65):
        raw = bytes(rng.randrange(256) for _ in range(n))
        ids = make_ids(raw)
        with torch.no_grad():
            logits, (receipt,), _ = m(ids, audit=True)
        assert logits.shape == (1, n, 256)
        assert torch.isfinite(logits).all()
        receipt.verify(raw[:-1])


def test_12_order_sensitive_token_representation_is_learnable():
    m = AdaptiveByteTokenGRU()
    a = m.byte_emb(torch.tensor([97, 98]))
    b = m.byte_emb(torch.tensor([98, 97]))
    # At initialization the token pool is a mean; source code supplies
    # learnable recency without encoding lexical segmentation rules.
    assert torch.allclose(m._pool([a[0:1], a[1:2]]), m._pool([b[0:1], b[1:2]]))
    with torch.no_grad():
        m.position_recency.fill_(1.0)
    assert not torch.allclose(m._pool([a[0:1], a[1:2]]), m._pool([b[0:1], b[1:2]]))
    with torch.no_grad():
        m.position_recency.zero_()
    x = make_ids(b'abcdef0123456789')
    raw = b'abcdef0123456789'
    targets = torch.tensor(list(raw), dtype=torch.long)
    out = m(x, mode='surrogate', force='token')
    F.cross_entropy(out[0], targets).backward()
    assert m.position_recency.grad is not None
    assert m.position_recency.grad.abs().sum().item() > 0


def test_13_representability_gate_can_condition_on_observed_input_not_gold():
    """Analytic weights demonstrate G2 *representability*, not learned skill."""
    torch.manual_seed(60)
    model = AdaptiveByteTokenGRU(max_token_bytes=4).eval()
    with torch.no_grad():
        model.mode_gate.weight.zero_()
        model.mode_gate.bias.zero_()
        # Controller's current-observed-byte coordinate is at hidden offset.
        model.mode_gate.weight[0, 110] = 3.0
        model.byte_emb.weight[ord('A'), 0] = 2.0
        model.byte_emb.weight[ord('B'), 0] = -2.0
        a, (ra,), _ = model(make_ids(b'A' * 20), audit=True)
        b, (rb,), _ = model(make_ids(b'B' * 20), audit=True)
    assert set(ra.decisions) == {'BYTE'}
    assert set(rb.decisions) == {'TOKEN'}
    assert ra.recurrent_updates == 19
    assert rb.recurrent_updates == 4  # 19 observed with token max length=4
    assert a.shape == b.shape == (1, 20, 256)
    assert not torch.equal(a, b)
