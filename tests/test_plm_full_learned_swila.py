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
    loss = torch.nn.functional.cross_entropy(output["action_logits"], target)
    loss = loss + 0.01 * output["load_loss"]
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
    ]
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in required)


def test_history_pair_has_identical_current_surface_but_different_target():
    left, right = make_history_pair(__import__("random").Random(19))
    assert torch.equal(left.text[-1], right.text[-1])
    assert torch.equal(left.image[-1], right.image[-1])
    assert torch.equal(left.audio[-1], right.audio[-1])
    assert left.action != right.action
