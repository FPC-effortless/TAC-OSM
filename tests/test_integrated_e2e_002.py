import inspect
import random
import pytest

torch = pytest.importorskip("torch")

from tac_osm.integrated_e2e import IntegratedE2EModel
from scripts.run_integrated_e2e_002 import ALL_COMBOS, HELDOUT, TRAIN_COMBOS, sample_episode

def test_training_and_holdout_compositions_are_disjoint():
    assert set(HELDOUT).isdisjoint(TRAIN_COMBOS)
    assert all(i < j for _, i, j in HELDOUT + TRAIN_COMBOS)
    assert all((op, j, i) not in TRAIN_COMBOS for op, i, j in HELDOUT)

def test_q1_and_q2_target_the_entities_whose_payloads_define_their_answers():
    episode = sample_episode(random.Random(900001), combo1=("xor", 0, 4), combo2=("and", 1, 9))
    observations, q1, q2, payload = episode
    entities = [row[0] for row in observations]
    assert q1[0] == entities[0]
    assert q2[0] == entities[1]
    assert q1[0] != q2[0]
    for q in (q1, q2):
        entity, i, j, op, answer = q
        a, b = payload[entity][i], payload[entity][j]
        expected = {"xor": a ^ b, "and": a & b, "or": a | b, "xnor": 1 - (a ^ b)}[op]
        assert answer == expected

def test_q2_cannot_silently_refer_to_q1_entity():
    episode = sample_episode(random.Random(900002), combo1=("xor", 0, 4), combo2=("and", 1, 9))
    observations, q1, q2, payload = episode
    assert q2[0] != q1[0]
    original = q2[4]
    mutated = dict(payload)
    mutated[q2[0]] = list(mutated[q2[0]])
    mutated[q2[0]][q2[1]] ^= 1
    a, b = mutated[q2[0]][q2[1]], mutated[q2[0]][q2[2]]
    changed = {"xor": a ^ b, "and": a & b, "or": a | b, "xnor": 1 - (a ^ b)}[q2[3]]
    assert changed != original

def test_query_interface_has_no_pre_action_answer_or_outcome():
    names = list(inspect.signature(IntegratedE2EModel.query).parameters)
    assert names == ["self", "memory", "entity", "i", "j", "op"]

def test_holdout_is_cross_modal():
    def modality(bit):
        return 0 if bit < 4 else 1 if bit < 8 else 2
    assert all(modality(i) != modality(j) for _, i, j in HELDOUT)

def test_registered_query_space_is_partitioned():
    assert set(ALL_COMBOS) == set(TRAIN_COMBOS) | set(HELDOUT)
    assert not (set(TRAIN_COMBOS) & set(HELDOUT))
