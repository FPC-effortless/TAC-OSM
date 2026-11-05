"""Engineering-only tests for CSL-001. Does not exercise authentic FIT/DEV."""
import math
import pytest
import torch
from tac_osm.csl001 import CSL001, CSLConfig, _bounded_scan, theoretical_weight_bytes


def model():
    torch.manual_seed(11)
    return CSL001().eval()


def test_model_shape_parameter_budget_and_rungs():
    m = model()
    assert 95_000 <= m.parameter_count() <= 105_000, m.parameter_count()
    x = torch.randint(0, 256, (2, 9), dtype=torch.long)
    with torch.no_grad():
        o, state = m(x, all_rungs=True)
    assert list(o) == [2, 4, 8]
    assert all(o[k]['token_logits'].shape == (2, 9, 256) for k in o)
    assert all(o[k]['field_logits'].shape == (2, 9, 16) for k in o)
    assert all(o[k]['tool_logits'].shape == (2, 17) for k in o)
    assert all(o[k]['accept_logit'].shape == (2, 1) for k in o)
    assert len(state.layers) == 8
    assert all(s.fast.dtype == torch.float32 for s in state.layers)


def test_depth_slice_identical_to_full_rung_output():
    m = model()
    x = torch.randint(0, 256, (2, 8))
    with torch.no_grad():
        full, _ = m(x, all_rungs=True)
        for rung in (2, 4, 8):
            chosen, _ = m(x, depth=rung)
            for key in full[rung]:
                torch.testing.assert_close(full[rung][key], chosen[rung][key], rtol=0, atol=0)


def test_no_future_token_leakage():
    m = model()
    x = torch.randint(0, 256, (2, 10))
    y = x.clone()
    y[:, 7:] = (y[:, 7:] + 103) % 256
    with torch.no_grad():
        a, _ = m(x, all_rungs=True)
        b, _ = m(y, all_rungs=True)
    for depth in (2, 4, 8):
        torch.testing.assert_close(a[depth]['token_logits'][:, :7], b[depth]['token_logits'][:, :7], rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(a[depth]['field_logits'][:, :7], b[depth]['field_logits'][:, :7], rtol=1e-5, atol=1e-5)


def test_streaming_state_equivalence_and_entity_isolation():
    m = model()
    x = torch.randint(0, 256, (2, 85))
    with torch.no_grad():
        full, _ = m(x, all_rungs=True)
        state = None
        parts = {r: [] for r in (2, 4, 8)}
        for start, end in ((0, 7), (7, 31), (31, 63), (63, 85)):
            out, state = m(x[:, start:end], state=state, all_rungs=True, stream_ids=("entity-a", "entity-b"))
            for r in parts:
                parts[r].append(out[r]['token_logits'])
        for r in parts:
            torch.testing.assert_close(full[r]['token_logits'], torch.cat(parts[r], dim=1), rtol=4e-4, atol=4e-4)
    # Missing an explicit reset must not silently be accepted as a new entity.
    # Model API cannot authenticate an identity; per-entity state assignment is caller-owned.
    for s in state.layers:
        assert s.fast.shape[0] == x.shape[0]


def test_scan_matches_serial_reference_and_has_gradients():
    torch.manual_seed(4)
    z = torch.randn(2, 91, 4, 16, requires_grad=True) / 6
    initial = torch.randn(2, 4, 16) / 8
    decay = torch.linspace(0.85, 0.99, 64).reshape(4, 16)
    out, ending = _bounded_scan(z, initial, decay, 32)
    s = initial
    ref = []
    for i in range(z.shape[1]):
        s = decay * s + (1-decay) * z[:, i]
        ref.append(s)
    torch.testing.assert_close(out, torch.stack(ref, dim=1), rtol=2e-5, atol=2e-5)
    torch.testing.assert_close(ending, s, rtol=2e-5, atol=2e-5)
    out.square().mean().backward()
    assert z.grad_fn is not None


def test_finite_backward_and_loss_at_all_rungs():
    m = model().train()
    x = torch.randint(0, 256, (2, 13))
    loss = m.nested_next_token_loss(x)
    assert math.isfinite(loss.item())
    loss.backward()
    assert m.embedding.weight.grad is not None
    assert all(b.key.weight.grad is not None for b in m.blocks)
    assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)
    # Specialized heads are not in the text-only loss, explicitly not trained.
    assert m.accept_head.weight.grad is None


def test_reject_unregistered_depth_and_wrong_shape():
    m = model()
    with pytest.raises(ValueError):
        m(torch.randint(0, 256, (1, 2)), depth=6)
    with pytest.raises(ValueError):
        m(torch.ones(1, 2))
    with pytest.raises(ValueError):
        CSLConfig(rungs=(2, 8), layers=6)
    assert theoretical_weight_bytes(10, 2.125) == 3


def test_ablation_in_memory_state_is_measurable_but_not_quality_proof():
    m = model()
    x = torch.randint(0, 256, (1, 20))
    with torch.no_grad():
        one, state = m(x[:, :10], depth=2, stream_ids=("entity-a",))
        withmem, _ = m(x[:, 10:], depth=2, state=state, stream_ids=("entity-a",))
        reset, _ = m(x[:, 10:], depth=2)
    assert not torch.allclose(withmem[2]['token_logits'], reset[2]['token_logits'], atol=1e-7)



def test_wrong_entity_carry_is_rejected():
    m = model()
    x = torch.randint(0, 256, (1, 5))
    with torch.no_grad():
        _, state = m(x, stream_ids=("caller-1",))
        with pytest.raises(ValueError):
            m(x, state=state, stream_ids=("caller-2",))
        with pytest.raises(ValueError):
            m(x, state=state)
        with pytest.raises(ValueError):
            m(x, stream_ids=("reused", "reused"))


def test_depth_allocates_only_selected_state_and_rejects_switch_with_stale_cache():
    m = model()
    x = torch.randint(0, 256, (1, 5))
    with torch.no_grad():
        _, s2 = m(x, depth=2, stream_ids=("entity",))
        _, s4 = m(x, depth=4, stream_ids=("entity",))
        _, s8 = m(x, depth=8, stream_ids=("entity",))
        assert (len(s2.layers), len(s4.layers), len(s8.layers)) == (2, 4, 8)
        with pytest.raises(ValueError):
            m(x, depth=4, state=s2, stream_ids=("entity",))


def test_gold_supervised_task_heads_all_receive_gradients():
    m = model().train()
    x = torch.randint(0, 256, (2, 7))
    loss = m.supervised_task_loss(
        x, tool_labels=torch.tensor([0, 3]),
        field_labels=torch.randint(0, 16, (2, 7)),
        accept_labels=torch.tensor([0.0, 1.0]),
    )
    loss.backward()
    assert m.accept_head.weight.grad is not None
    assert m.tool_head.weight.grad is not None
    assert m.field_head.weight.grad is not None
    assert m.blocks[0].key.weight.grad is not None
    with pytest.raises(ValueError):
        m.supervised_task_loss(x, torch.tensor([0, 1]), torch.zeros_like(x), torch.tensor([0.2, 1.0]))


def test_twenty_layer_ladder_candidate_is_instantiable_not_trained():
    from tac_osm.csl001 import csl20_config
    m = CSL001(csl20_config()).eval()
    assert 95_000 <= m.parameter_count() <= 105_000
    with torch.no_grad():
        outputs, state = m(torch.randint(0, 256, (1, 6)), all_rungs=True)
    assert list(outputs) == [2, 8, 20]
    assert len(state.layers) == 20
