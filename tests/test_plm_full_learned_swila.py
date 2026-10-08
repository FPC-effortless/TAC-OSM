from __future__ import annotations

import torch

from tac_osm.plm_full_benchmark import (
    ACTION_COUNT,
    make_history_pair,
    sample_episode,
)
from tac_osm.plm_full_learned import (
    FullLearnedPLM,
    FullPLMConfig,
    full_plm_parameter_count,
)


def _small_model() -> FullLearnedPLM:
    return FullLearnedPLM(
        FullPLMConfig(
            vocab_size=32,
            d_model=48,
            num_heads=4,
            num_experts=6,
            fast_experts=2,
            medium_experts=2,
            slow_experts=2,
            cdl_top_k=2,
            casm_operators=4,
            casm_depth=2,
            action_count=ACTION_COUNT,
        )
    )


def test_full_plm_has_no_fixed_operator_table():
    model = _small_model()
    assert len(model.casm.operators) == 4
    assert all(not hasattr(op, "fn") for op in model.casm.operators)
    assert full_plm_parameter_count(model) > 0


def test_swilamtsk_state_is_fixed_and_timescale_partitioned():
    model = _small_model()
    state = model.mtsk.initial_state(2, torch.device("cpu"), torch.float32)
    assert state.shape == (2, 4, 6, 12, 12)
    decay = torch.sigmoid(model.mtsk.decay_logits).detach().tolist()
    assert len(decay) == 6
    assert max(decay[:2]) < min(decay[2:4]) < min(decay[4:])


def test_end_to_end_gradient_reaches_all_major_subsystems():
    torch.manual_seed(0)
    model = _small_model()
    episode = sample_episode(__import__("random").Random(7))
    output = model(**episode.batch)
    target = torch.tensor([episode.action])
    action_loss = torch.nn.functional.cross_entropy(output["action_logits"], target)

    _next_state, obs, evidence, reward, _ = __import__(
        "tac_osm.plm_full_benchmark", fromlist=["step_environment"]
    ).step_environment(episode, episode.action)
    post_z = model.representation(
        obs[0].unsqueeze(0), obs[1].unsqueeze(0), obs[2].unsqueeze(0)
    )
    predicted_next = model.osm.predict(
        output["query"],
        target,
        torch.tensor([evidence], dtype=output["action_logits"].dtype),
    )
    world_loss = torch.nn.functional.mse_loss(predicted_next, post_z.detach())

    step = model.verify_and_update(
        output,
        action=target,
        outcome=torch.tensor([evidence], dtype=output["action_logits"].dtype),
        post_text=obs[0].unsqueeze(0),
        post_image=obs[1].unsqueeze(0),
        post_audio=obs[2].unsqueeze(0),
    )
    verifier_target = torch.tensor([float(reward)], dtype=step.verifier_logit.dtype)
    verifier_loss = torch.nn.functional.binary_cross_entropy_with_logits(
        step.verifier_logit, verifier_target
    )
    repair_loss = torch.nn.functional.cross_entropy(step.repair_logits, target)

    # The production training objective supplies separate losses for the
    # action, OSM, verifier, and repair paths; this test verifies that each
    # registered subsystem is actually reachable under that integrated graph.
    loss = (
        action_loss
        + 0.10 * output["load_loss"]
        + 0.10 * world_loss
        + 0.10 * verifier_loss
        + 0.10 * repair_loss
        + 0.01 * step.next_memory_state.square().mean()
    )
    loss.backward()

    required = [
        model.representation.text.embedding.weight,
        model.mtsk.q_proj.weight,
        model.cdl.score[0].weight,
        model.casm.selector[0].weight,
        model.osm.transition[0].weight,
        model.verifier.net[0].weight,
        model.repair.net[0].weight,
        model.action_head[0].weight,
        model.mtsk.write_proj.weight,
    ]
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in required)


def test_history_pair_has_identical_current_surface_but_different_target():
    left, right = make_history_pair(__import__("random").Random(19))
    assert torch.equal(left.text[-1], right.text[-1])
    assert torch.equal(left.image[-1], right.image[-1])
    assert torch.equal(left.audio[-1], right.audio[-1])
    assert left.action != right.action



def test_verifier_contract_uses_continuous_outcome_not_binary_success():
    from tac_osm.plm_full_learned import FullPLMConfig, LearnedVerifier

    cfg = FullPLMConfig(d_model=48, num_heads=4, action_count=5)
    verifier = LearnedVerifier(cfg)
    expected_in = 2 * cfg.d_model + cfg.action_count + 1
    first = verifier.net[0]
    assert first.in_features == expected_in
    assert first.in_features != 2 * cfg.d_model + 2


def test_verified_write_is_gated_and_state_persists():
    torch.manual_seed(3)
    model = _small_model()
    state = model.mtsk.initial_state(1, torch.device("cpu"), torch.float32)
    context = torch.randn(1, 48)
    action = torch.tensor([1])
    outcome = torch.tensor([-0.25])

    blocked = model.mtsk.commit_verified(
        state, context, action, outcome, torch.zeros(1)
    )
    admitted = model.mtsk.commit_verified(
        state, context, action, outcome, torch.ones(1)
    )

    assert torch.equal(blocked, state)
    assert not torch.equal(admitted, state)

def test_physics_prior_loss_is_finite_and_differentiable():
    torch.manual_seed(11)
    model = _small_model()
    episode = sample_episode(__import__("random").Random(31))
    output = model(**episode.batch)
    loss = model.physics_prior_loss(output["z_seq"], dt=0.08)
    assert torch.isfinite(loss)
    loss.backward()
    assert model.physics_latent[0].weight.grad is not None
    assert torch.isfinite(model.physics_latent[0].weight.grad).all()


def test_verified_write_persists_into_later_forward():
    torch.manual_seed(13)
    model = _small_model()
    state = model.mtsk.initial_state(1, torch.device("cpu"), torch.float32)
    context = torch.randn(1, 48)
    action = torch.tensor([1])
    outcome = torch.tensor([-0.25])
    admitted = model.mtsk.commit_verified(
        state, context, action, outcome, torch.ones(1)
    )
    episode = sample_episode(__import__("random").Random(23))
    baseline = model(**episode.batch)
    carried = model(
        episode.batch["text"],
        episode.batch["image"],
        episode.batch["audio"],
        memory_state=admitted,
    )
    assert not torch.allclose(
        baseline["memory_state"], carried["memory_state"]
    )


def test_benchmark_generator_hash_matches_file():
    import hashlib
    from tac_osm import plm_full_benchmark

    expected = hashlib.sha256(
        open(plm_full_benchmark.__file__, "rb").read()
    ).hexdigest()
    assert plm_full_benchmark.generator_hash() == expected
