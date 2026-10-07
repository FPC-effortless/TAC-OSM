import inspect
import random

import pytest

torch = pytest.importorskip("torch")
import torch as th

from tac_osm.integrated_e2e_005 import FunctionalConfig, FunctionalMultimodalPLM
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
    ep = sample_episode(
        random.Random(5005),
        combo1=HELDOUT[0],
        combo2=HELDOUT[2],
    )
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
    ep = sample_episode(
        random.Random(7005),
        combo1=HELDOUT[0],
        combo2=HELDOUT[1],
    )
    _, _, q2, payload = ep
    original = q2[4]

    def truth(a, b):
        return {
            "xor": a ^ b,
            "and": a & b,
            "or": a | b,
            "xnor": 1 - (a ^ b),
        }[q2[3]]

    for a in (0, 1):
        for b in (0, 1):
            if truth(a, b) != original:
                changed = {k: list(v) for k, v in payload.items()}
                changed[q2[0]][q2[1]] = a
                changed[q2[0]][q2[2]] = b
                assert truth(
                    changed[q2[0]][q2[1]],
                    changed[q2[0]][q2[2]],
                ) != original
                return
    raise AssertionError("could not construct a target-payload mutation that changes the answer")


def test_fingerprint_changes_when_observation_changes():
    episodes = sample_evaluation_episodes(random.Random(8005), 8)
    before = episode_fingerprint(episodes)
    row = episodes[0][0][0]
    episodes[0][0][0] = (
        row[0],
        row[1],
        row[2] + 0.01,
        row[3],
    )
    after = episode_fingerprint(episodes)
    assert before != after


def test_model_query_has_no_target_bearing_arguments():
    assert list(inspect.signature(FunctionalMultimodalPLM.query).parameters) == [
        "self", "memory", "entity", "i", "j", "op"
    ]
    assert "answer" not in inspect.getsource(FunctionalMultimodalPLM.query)


def test_model_has_no_label_bearing_episode_forward():
    assert not hasattr(FunctionalMultimodalPLM, "forward_episode")
    source = inspect.getsource(FunctionalMultimodalPLM)
    assert "def forward_episode" not in source


def test_post_action_update_requires_feedback_separately():
    assert list(inspect.signature(FunctionalMultimodalPLM.post_action_update).parameters) == [
        "self", "memory", "query_result", "outcome"
    ]
    source = inspect.getsource(FunctionalMultimodalPLM.post_action_update)
    assert "q1" not in source and "q2" not in source
    assert "answer" not in source


def test_final_action_loss_has_full_multimodal_gradient_surface():
    rng = random.Random(9005)
    episodes = [sample_episode(rng) for _ in range(2)]
    model = FunctionalMultimodalPLM()

    from scripts.run_integrated_e2e_005 import build_batch, write_observations

    batch = build_batch(episodes)
    memory = write_observations(model, batch)
    entity, i, j, op = batch["q1"]
    out = model.query(memory, entity, i, j, op)
    loss = th.nn.functional.cross_entropy(out["logits"], batch["y1"])
    loss.backward()

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


def test_action_output_does_not_depend_on_target_label():
    rng = random.Random(9010)
    ep = sample_episode(rng, combo1=HELDOUT[0], combo2=HELDOUT[1])
    model = FunctionalMultimodalPLM()
    from scripts.run_integrated_e2e_005 import build_batch, write_observations

    batch = build_batch([ep])
    memory = write_observations(model, batch)
    entity, i, j, op = batch["q2"]

    out = model.query(memory, entity, i, j, op)
    logits_a = out["logits"].detach().clone()

    opposite_target = 1 - ep[2][4]
    loss_b = th.nn.functional.cross_entropy(
        logits_a,
        th.tensor([opposite_target]),
    )
    assert th.isfinite(loss_b)


def test_post_action_feedback_cannot_modify_another_entity_slot():
    model = FunctionalMultimodalPLM()
    memory = model.state.initial(1, th.device("cpu"))
    entity_a = th.tensor([2])
    entity_b = th.tensor([3])
    z = th.randn(1, model.config.hidden_dim)
    memory, _ = model.state.write(memory, z, entity_a)
    before_b, _ = model.state.read(memory, entity_b)
    query_result = {
        "entity": entity_a,
        "read": th.randn(1, model.config.hidden_dim),
        "action": th.tensor([0.9]),
    }
    updated, _ = model.post_action_update(memory, query_result, th.ones(1))
    after_b, _ = model.state.read(updated, entity_b)
    assert th.equal(before_b, after_b)


def test_training_harness_derives_environment_outcome_after_query():
    from scripts.run_integrated_e2e_005 import train_batch
    source = inspect.getsource(train_batch)
    first_query = source.index("model.query")
    first_outcome = source.index("environment_outcome")
    assert first_query < first_outcome


def test_single_episode_audio_has_explicit_channel_dimension():
    ep = sample_episode(random.Random(10005), combo1=HELDOUT[0], combo2=HELDOUT[1])
    _, _, _, audio = ep[0][0]
    model = FunctionalMultimodalPLM()
    encoded = model.audio(audio.unsqueeze(0).unsqueeze(0))
    assert encoded.shape == (1, model.config.hidden_dim // 2)


def test_registered_primary_threshold_is_read_from_contract():
    from scripts.run_integrated_e2e_005 import primary_threshold
    assert primary_threshold() == 0.80


def test_zero_outcome_does_not_change_memory():
    model = FunctionalMultimodalPLM()
    memory = model.state.initial(1, th.device("cpu"))
    query_result = {
        "entity": th.tensor([2]),
        "read": th.randn(1, model.config.hidden_dim),
        "action": th.tensor([0.2]),
    }
    before = memory.clone()
    after, _ = model.post_action_update(
        memory,
        query_result,
        th.zeros(1),
    )
    assert th.equal(before, after)


def test_fixed_casm_probability_to_logit_interface_preserves_binary_decision():
    model = FunctionalMultimodalPLM()
    state = th.zeros(2, model.config.hidden_dim)
    i = th.tensor([0, 0])
    j = th.tensor([1, 1])
    op = th.tensor([1, 1])  # AND
    # Force decoder to produce confidently different latent states.
    with th.no_grad():
        for p in model.casm.decoder.parameters():
            p.zero_()
        model.casm.decoder[-1].bias[0] = 8.0
        model.casm.decoder[-1].bias[1] = 8.0
    action, logits, _ = model.casm(state, i, j, op)
    assert th.all(action > 0.99)
    assert th.equal(logits.argmax(-1), th.ones(2, dtype=th.long))


def test_fixed_casm_logits_do_not_force_positive_class_for_zero_action():
    model = FunctionalMultimodalPLM()
    state = th.zeros(1, model.config.hidden_dim)
    i = th.tensor([0])
    j = th.tensor([1])
    op = th.tensor([1])  # AND
    with th.no_grad():
        for p in model.casm.decoder.parameters():
            p.zero_()
        model.casm.decoder[-1].bias[0] = -8.0
        model.casm.decoder[-1].bias[1] = -8.0
    action, logits, _ = model.casm(state, i, j, op)
    assert action.item() < 1e-5
    assert logits.argmax(-1).item() == 0


def test_legacy_default_capacity_remains_40_for_e2e_005_reproducibility():
    assert FunctionalConfig().hidden_dim == 40
