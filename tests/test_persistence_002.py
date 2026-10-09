"""Integrity gates for the new PERSISTENCE-002 scientific experiment."""
from dataclasses import replace

import pytest
import torch
import torch.nn.functional as F

from tac_osm.persistence_002 import PersistencePLM002
from tac_osm.persistence_002_benchmark import (
    DELAYS,
    assert_valid_pairs,
    make_paired_batch,
    pool_fingerprint,
)


def _batch(delay=4, seed=10):
    return make_paired_batch(
        torch.Generator().manual_seed(seed), pairs=5, delay=delay
    )


def test_contract_is_preregistered_and_runtime_parameters_pinned():
    from tac_osm.contract import load_contract
    from scripts.run_persistence_002 import preflight

    c = load_contract("TACOSM-PLM-PERSISTENCE-002")
    assert c.status == "pre-registered"
    assert c.check_consistency() == []
    assert c.h_levels == DELAYS
    assert len(c.arms) == 5
    preflight()


def test_all_nonzero_delays_have_identical_present_and_opposite_pasts():
    for delay in DELAYS:
        b = _batch(delay=delay)
        assert_valid_pairs(b)
        assert torch.equal(b.current[0][::2], b.current[0][1::2])
        assert torch.equal(b.current[1][::2], b.current[1][1::2])
        assert torch.equal(b.current[2][::2], b.current[2][1::2])
        assert not torch.equal(b.t0[0][::2], b.t0[0][1::2])
        assert torch.equal(b.target[::2], 1 - b.target[1::2])


def test_evaluation_generator_has_independent_fingerprints():
    assert pool_fingerprint(_batch(seed=11)) != pool_fingerprint(_batch(seed=12))


def test_incorrect_delay_count_fails_closed():
    with pytest.raises(AssertionError, match="step count"):
        assert_valid_pairs(replace(_batch(), delay=8))


def test_modified_distractor_inside_pair_is_rejected():
    b = _batch()
    altered = list(b.distractors)
    text, image, audio = altered[0]
    text = text.clone()
    text[1, 1] = (text[1, 1] - 11) ^ 1
    altered[0] = (text, image, audio)
    with pytest.raises(AssertionError, match="different later inputs"):
        assert_valid_pairs(replace(b, distractors=tuple(altered)))


def test_hidden_labels_do_not_enter_model_path():
    torch.manual_seed(5)
    b = _batch()
    model = PersistencePLM002()
    altered = replace(
        b, memory_bit=1 - b.memory_bit, target=1 - b.target
    )
    assert torch.equal(model(b).logits, model(altered).logits)


def test_learned_recurrent_write_receives_gradients_after_distractors():
    torch.manual_seed(6)
    model = PersistencePLM002()
    b = _batch(delay=4)
    loss = F.cross_entropy(model(b).logits, b.target)
    loss.backward()
    for gradient in (
        model.write_cell.weight_ih.grad,
        model.write_cell.weight_hh.grad,
    ):
        assert gradient is not None
        assert torch.isfinite(gradient).all()
        assert float(gradient.abs().sum()) > 0


def test_recurrent_cell_executes_every_distractor_and_changes_state():
    torch.manual_seed(7)
    model = PersistencePLM002()
    b = _batch(delay=8)
    called = []
    handle = model.write_cell.register_forward_hook(
        lambda _module, _inp, _out: called.append(1)
    )
    try:
        trace = model(b)
    finally:
        handle.remove()
    assert len(called) == 9
    assert trace.state_write_calls == 9
    assert len(trace.distractor_update_norms) == 8
    assert all(float(x) > 0.0001 for x in trace.distractor_update_norms)


def test_current_only_reset_is_exact_half_on_same_present_pairs():
    torch.manual_seed(8)
    b = _batch(delay=16)
    model = PersistencePLM002()
    trace = model(b)
    z = model.predict(torch.zeros_like(trace.state), trace.current_representation)
    pred = z.argmax(dim=-1)
    assert torch.equal(pred[::2], pred[1::2])
    assert (pred == b.target).float().mean().item() == 0.5


def test_oracle_and_pairwise_state_swap_are_benchmark_controls():
    b = _batch(delay=2)
    oracle = b.memory_bit ^ b.current_bit
    assert torch.equal(oracle, b.target)
    swaps = torch.arange(oracle.numel()).reshape(-1, 2).flip(1).flatten()
    assert torch.equal(b.memory_bit[swaps], 1 - b.memory_bit)
    assert torch.equal(b.current_bit[swaps], b.current_bit)
