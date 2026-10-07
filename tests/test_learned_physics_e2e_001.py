import json
import random

import torch

from tac_osm.learned_physics_e2e import (
    LearnedActionHead,
    LearnedPhysicsConfig,
    LearnedPhysicsE2E,
    physics_prior_loss,
)
from tac_osm.learned_physics_e2e_benchmark import (
    ACTION_COUNT,
    HISTORY,
    sample_episode,
    sample_history_pairs,
)


def test_benchmark_episode_shape_and_action_domain():
    episode = sample_episode(random.Random(1234))
    assert len(episode["observations"]) == HISTORY
    assert 0 <= int(episode["action"]) < ACTION_COUNT
    assert episode["observations"][-1][0].shape == (5,)


def test_history_pair_has_identical_current_multimodal_surface():
    left, right = sample_history_pairs(random.Random(8), 1)[0]
    for first, second in zip(left["observations"][-1], right["observations"][-1]):
        assert torch.equal(first, second)
    assert left["action"] != right["action"]


def test_model_is_generic_learned_computation():
    model = LearnedPhysicsE2E(LearnedPhysicsConfig())
    assert sum(parameter.numel() for parameter in model.parameters()) > 0
    assert not hasattr(model, "entity_ids")
    assert not hasattr(model, "operator_table")
    assert not hasattr(model, "fixed_executor")


def test_action_path_consumes_physical_state():
    head = LearnedActionHead(64, ACTION_COUNT)
    temporal = torch.randn(2, 64, requires_grad=True)
    physical = torch.randn(2, 8, requires_grad=True)
    goal = torch.randn(2, 2)
    logits = head(temporal, physical, goal)
    gradient = torch.autograd.grad(logits.sum(), physical)[0]
    assert torch.isfinite(gradient).all()
    assert float(gradient.abs().sum()) > 0.0


def test_physics_prior_uses_only_model_predictions():
    model = LearnedPhysicsE2E(LearnedPhysicsConfig(physics_weight=0.05))
    text = torch.randint(0, 32, (2, HISTORY, 5))
    image = torch.rand(2, HISTORY, 1, 32, 32)
    audio = torch.rand(2, HISTORY, 1, 96)
    goal = torch.rand(2, 2)
    logits, states = model(text, image, audio, goal)
    assert logits.shape == (2, ACTION_COUNT)
    assert states.shape == (2, HISTORY, 8)
    losses = physics_prior_loss(states, 0.08)
    assert set(losses) == {"total", "momentum", "energy", "kinematic"}
    assert torch.isfinite(losses["total"])


def test_contract_boundary_is_explicit():
    with open("contracts/TACOSM-PLM-LEARNED-PHYSICS-E2E-001.json", encoding="utf-8") as handle:
        contract = json.load(handle)
    assert contract["status"] == "pre-registered"
    assert contract["physics_prior"]["allowed_information"]
    assert contract["physics_prior"]["forbidden_information"]
    assert contract["protocol"]["model_selection"] == "none"
