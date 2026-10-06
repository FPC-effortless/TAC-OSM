import pytest

torch = pytest.importorskip("torch")
import random

from tac_osm.integrated_e2e_benchmark import (
    HELDOUT,
    TRAIN_COMBOS,
    sample_episode,
    validate_episode,
)


def test_corrected_generator_targets_distinct_query_entities():
    rng = random.Random(12345)
    for _ in range(200):
        ep = sample_episode(rng)
        assert ep[1][0] != ep[2][0]
        assert (ep[1][3], ep[1][1], ep[1][2]) in TRAIN_COMBOS
        assert (ep[2][3], ep[2][1], ep[2][2]) in TRAIN_COMBOS


def test_every_heldout_composition_is_excluded_from_training():
    assert all(combo not in TRAIN_COMBOS for combo in HELDOUT)


def test_corrected_generator_handles_each_registered_heldout_pair():
    rng = random.Random(20261002)
    for combo1 in HELDOUT:
        for combo2 in HELDOUT:
            if combo1 == combo2:
                continue
            ep = sample_episode(rng, combo1=combo1, combo2=combo2)
            validate_episode(ep)
            assert (ep[1][3], ep[1][1], ep[1][2]) == combo1
            assert (ep[2][3], ep[2][1], ep[2][2]) == combo2
            assert ep[1][0] != ep[2][0]

def test_query_answer_is_derived_from_named_entity_only():
    rng = random.Random(20261006)
    ep = sample_episode(rng, combo1=("xor", 0, 4), combo2=("and", 1, 9))
    payload = ep[3]
    for q in (ep[1], ep[2]):
        entity, i, j, op, answer = q
        a, b = payload[entity][i], payload[entity][j]
        expected = {"xor": a ^ b, "and": a & b, "or": a | b, "xnor": 1 - (a ^ b)}[op]
        assert answer == expected

def test_post_action_feedback_is_not_the_target_answer():
    from pathlib import Path
    source = (Path(__file__).resolve().parents[1] / "src" / "tac_osm" / "integrated_e2e.py").read_text()
    assert 'action1_hard = out1["logits"].argmax(dim=-1)' in source
    assert 'outcome1 = (action1_hard == q1[-1]).to(out1["logits"].dtype)' in source
    assert 'post_action_update(memory, out1, outcome1)' in source
    assert 'post_action_update(memory, out1, q1[-1])' not in source
