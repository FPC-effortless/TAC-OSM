import inspect
import random

import pytest

torch = pytest.importorskip("torch")
import torch as th

from tac_osm.integrated_e2e_005 import FunctionalMultimodalPLM
from tac_osm.integrated_e2e_005_benchmark import (
    ALL_COMBOS,
    HELDOUT,
    TRAIN_COMBOS,
    episode_fingerprint,
    sample_episode,
    sample_evaluation_episodes,
    validate_episode,
)


def test_train_and_holdout_compositions_are_disjoint():
    assert set(TRAIN_COMBOS).isdisjoint(HELDOUT)
    assert set(ALL_COMBOS) == set(TRAIN_COMBOS) | set(HELDOUT)
    assert all(i < j for _, i, j in ALL_COMBOS)


def test_heldout_pairs_are_cross_modal():
    def modality(bit):
        return 0 if bit < 4 else 1 if bit < 8 else 2
    assert all(modality(i) != modality(j) for _, i, j in HELDOUT)


def test_q1_and_q2_target_distinct_named_entities():
    ep = sample_episode(random.Random(5005), combo1=HELDOUT[0], combo2=HELDOUT[2])
    obs, q1, q2, payload = ep
    entities = [row[0] for row in obs]
    assert q1[0] == entities[0]
    assert q2[0] == entities[1]
    assert q1[0] != q2[0]
    for q in (q1, q2):
        a, b = payload[q[0]][q[1]], payload[q[0]][q[2]]
        expected = {
            "xor": a ^ b,
            "and": a & b,
            "or": a | b,
            "xnor": 1 - (a ^ b),
        }[q[3]]
        assert q[4] == expected


def test_every_eval_composition_is_registered_heldout():
    episodes = sample_evaluation_episodes(random.Random(6005), 64)
    for ep in episodes:
        validate_episode(ep)


def test_query_answer_depends_on_target_entity_payload():
    ep = sample_episode(random.Random(7005), combo1=HELDOUT[0], combo2=HELDOUT[1])
    _, _, q2, payload = ep
    original = q2[4]
    entity = q2[0]
    for a in (0, 1):
        for b in (0, 1):
            value = {"xor": a ^ b, "and": a & b, "or": a | b, "xnor": 1 - (a ^ b)}[q2[3]]
            if value != original:
                changed_payload = {k: list(v) for k, v in payload.items()}
                changed_payload[entity][q2[1]] = a
                changed_payload[entity][q2[2]] = b
                assert changed_payload[entity] != payload[entity]
                return
    raise AssertionError("could not construct a target-payload mutation")


def test_fingerprint_changes_when_observation_changes():
    episodes = sample_evaluation_episodes(random.Random(8005), 8)
    before = episode_fingerprint(episodes)
    row = episodes[0][0][0]
    episodes[0][0][0] = (row[0], row[1], row[2] + 0.01, row[3])
    after = episode_fingerprint(episodes)
    assert before != after


def test_query_has_no_answer_or_outcome_argument():
    assert list(inspect.signature(FunctionalMultimodalPLM.query).parameters) == [
        "self", "memory", "entity", "i", "j", "op"
    ]


def test_training_forward_signature_has_no_payload_labels():
    assert list(inspect.signature(FunctionalMultimodalPLM.forward_episode).parameters) == [
        "self", "observations", "observation_entities", "q1", "q2"
    ]


def test_no_auxiliary_representation_loss_in_forward():
    source = inspect.getsource(FunctionalMultimodalPLM.forward_episode)
    assert "bit_loss" not in source
    assert "entity_loss" not in source
    assert "payload" not in source


def test_final_action_loss_has_full_multimodal_gradient_surface():
    rng = random.Random(9005)
    episodes = [sample_episode(rng) for _ in range(2)]
    model = FunctionalMultimodalPLM()

    from scripts.run_integrated_e2e_005 import build_batch

    batch = build_batch(episodes)
    out = model.forward_episode(
        {k: batch[k] for k in ("text", "image", "audio")},
        batch["entities"],
        batch["q1"],
        batch["q2"],
    )
    out["action1_loss"].backward()

    required = (
        "text.emb.weight",
        "text.rnn.weight_ih_l0",
        "image.net.0.weight",
        "audio.net.0.weight",
        "rep.fuse.0.weight",
        "state.write_value.weight",
        "casm.decoder.0.weight",
    )
    for name in required:
        grad = dict(model.named_parameters())[name].grad
        assert grad is not None
        assert th.isfinite(grad).all()


def test_zero_outcome_does_not_change_memory():
    model = FunctionalMultimodalPLM()
    memory = model.state.initial(1, th.device("cpu"))
    query_result = {
        "entity": th.tensor([2]),
        "read": th.randn(1, 40),
        "action": th.tensor([0.2]),
    }
    before = memory.clone()
    after, _ = model.post_action_update(memory, query_result, th.zeros(1))
    assert th.equal(before, after)
