"""Nontrivial FIFO and typed-boundary checks for batched observation ingestion."""
import pytest
import torch
from tac_osm.development.semantic_state import SemanticStateAgent
from tac_osm.development.semantic_runtime import observe_many


def model():
    torch.set_num_threads(1)
    m = SemanticStateAgent(seed=65).double().eval()
    # A controlled gate classifies all six types; the tested mechanism is FIFO,
    # not the scientific benefit of the admission labels.
    with torch.no_grad():
        m.gate[0].weight.zero_(); m.gate[0].bias.zero_()
        m.gate[0].weight[:6].copy_(torch.eye(6, dtype=torch.float64))
        m.gate[2].weight.fill_(1.); m.gate[2].weight[:, 5] = -2.
        m.gate[2].bias.zero_()
    return m


@pytest.mark.parametrize('batch', [1, 3])
def test_masked_chunking_and_overflow_match_every_sequential_write(batch):
    m = model(); g = torch.Generator().manual_seed(6650909)
    keys = torch.randn(batch, 2, 16, generator=g, dtype=torch.float64)
    events = torch.randn(batch, 150, 46, generator=g, dtype=torch.float64)
    types = torch.arange(150) % 6
    events[:, :, :6] = torch.nn.functional.one_hot(types, 6)
    present = torch.rand(batch, 150, generator=g) > .1
    sequential, batched = m.initial(keys), m.initial(keys)
    for left, right in ((0, 9), (9, 73), (73, 150)):
        before = {k: v.clone() for k, v in batched.items()}
        old = batched
        batched = observe_many(m, batched, events[:, left:right], present[:, left:right])
        assert all(torch.equal(old[k], v) for k, v in before.items()), 'Input state was mutated'
        for tick in range(left, right):
            sequential = m.observe(sequential, events[:, tick], present[:, tick])
        for key in sequential:
            torch.testing.assert_close(batched[key], sequential[key], atol=0, rtol=0)
    assert (batched['present'].sum(-1) == 64).all()


def test_rejected_and_empty_chunks_preserve_state_without_copying():
    m = model(); state = m.initial(torch.ones(1, 2, 16, dtype=torch.float64))
    events = torch.zeros(1, 32, 46, dtype=torch.float64); events[:, :, 5] = 1
    assert observe_many(m, state, events) is state
    assert observe_many(m, state, events[:, :0]) is state


def test_invalid_public_types_cannot_silently_change_admission():
    m = model(); state = m.initial(torch.ones(1, 2, 16, dtype=torch.float64))
    events = torch.zeros(1, 2, 46, dtype=torch.float64)
    with pytest.raises(ValueError, match='one-hot'):
        observe_many(m, state, events)
    # Absent padding is allowed; it cannot enter storage.
    assert observe_many(m, state, events, torch.zeros(1, 2, dtype=torch.bool)) is state
