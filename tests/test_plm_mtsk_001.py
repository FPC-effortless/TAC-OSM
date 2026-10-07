import random

import torch

from tac_osm.contract import load_contract
from tac_osm.mtsk import MTSKEntityState, SingleAdaptiveTimescaleState
from tac_osm.mtsk_001_benchmark import (
    EVAL_DELAYS,
    EVAL_EPISODES_PER_DELAY,
    MASK_TOKEN,
    sample_evaluation_episodes,
    sample_sequence,
    target_balance,
)


def test_contract_is_preregistered():
    c = load_contract("TACOSM-PLM-MTSK-001")
    assert c.check_consistency() == []
    assert c.seeds == (0, 1, 2, 3, 4)
    assert c.steps == 200
    assert c.eval_steps == 200


def test_masking_removes_slow_bit_after_t0():
    ep = sample_sequence(random.Random(42), 8, slow=1, final_fast=0)
    assert int(ep["observations"][0]["text"][2]) == 21
    for row in ep["observations"][1:]:
        assert int(row["text"][2]) == MASK_TOKEN
        assert int(row["text"][3]) in (20, 21)


def test_evaluation_balance():
    eps = sample_evaluation_episodes(random.Random(123))
    counts = target_balance(eps)
    assert all(all(v == 50 for v in row.values()) for row in counts.values())
    assert len(eps) == len(EVAL_DELAYS) * EVAL_EPISODES_PER_DELAY


def test_parameter_matched_state_shapes_and_order():
    a = SingleAdaptiveTimescaleState(60, 1, 3)
    m = MTSKEntityState(60, 1, 3)
    assert sum(p.numel() for p in a.parameters()) == sum(p.numel() for p in m.parameters())
    assert bool(m.ordered_decays()[0] <= m.ordered_decays()[1] <= m.ordered_decays()[2])
    memory = m.initial(4, torch.device("cpu"))
    z = torch.randn(4, 60)
    updated, _ = m.write(memory, z, torch.zeros(4, dtype=torch.long))
    assert updated.shape == (4, 1, 3, 20)
    read, _ = m.read(updated, torch.zeros(4, dtype=torch.long))
    assert read.shape == (4, 60)
