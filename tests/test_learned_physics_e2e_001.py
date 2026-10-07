import json
import random

import torch

from tac_osm.learned_physics_e2e import (
    LearnedPhysicsConfig,
    LearnedPhysicsE2E,
    hamiltonian_prior_loss,
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


def test_history_pair_has_identical_current_multimodal_surface():
    left, right = sample_history_pairs(random.Random(8), 1)[0]
    for first, second in zip(left["observations"][-1], right["observations"][-1]):
        assert torch.equal(first, second)
    assert left["action"] != right["action"]


def test_model_has_no_task_specific_physical_ontology():
    model = LearnedPhysicsE2E(LearnedPhysicsConfig())
    assert sum(parameter.numel() for parameter in model.parameters()) > 0
    assert not hasattr(model, "entity_ids")
    assert not hasattr(model, "operator_table")
    assert not hasattr(model, "fixed_executor")
    assert not hasattr(model, "particle_count")


def test_hamiltonian_prior_is_on_shared_learned_latent():
    model = LearnedPhysicsE2E(LearnedPhysicsConfig())
    text = torch.randint(0, 32, (2, HISTORY, 5))
    image = torch.rand(2, HISTORY, 1, 32, 32)
    audio = torch.rand(2, HISTORY, 1, 96)
    goal = torch.rand(2, 2)
    logits, canonical, energy = model(text, image, audio, goal)
    assert logits.shape == (2, ACTION_COUNT)
    assert canonical.shape == (2, HISTORY, 64)
    assert energy.shape == (2, HISTORY)
    canonical.retain_grad()
    loss = hamiltonian_prior_loss(canonical, energy, 0.08)["total"]
    assert torch.isfinite(loss)
    gradient = torch.autograd.grad(loss, canonical, retain_graph=True)[0]
    assert torch.isfinite(gradient).all()


def test_contract_prior_boundary_is_explicit():
    with open("contracts/TACOSM-PLM-LEARNED-PHYSICS-E2E-001.json", encoding="utf-8") as handle:
        contract = json.load(handle)
    assert contract["status"] == "pre-registered"
    assert contract["physics_prior"]["allowed_information"]
    assert contract["physics_prior"]["forbidden_information"]


def test_audio_rendering_is_invariant_to_particle_order():
    from tac_osm.learned_physics_e2e_benchmark import PhysicalState, _render_audio
    state = PhysicalState(
        position=torch.tensor([[0.30, 0.35], [0.70, 0.65]]),
        velocity=torch.tensor([[0.12, 0.04], [-0.18, -0.07]]),
    )
    swapped = PhysicalState(state.position.flip(0), state.velocity.flip(0))
    first = _render_audio(state, random.Random(5), deterministic=True)
    second = _render_audio(swapped, random.Random(5), deterministic=True)
    assert torch.equal(first, second)


def test_action_path_consumes_canonical_latent():
    from tac_osm.learned_physics_e2e import LearnedActionHead
    head = LearnedActionHead(64, ACTION_COUNT)
    state = torch.randn(2, 64, requires_grad=True)
    canonical = torch.randn(2, 64, requires_grad=True)
    goal = torch.randn(2, 2)
    logits = head(state, canonical, goal)
    grad = torch.autograd.grad(logits.sum(), canonical)[0]
    assert torch.isfinite(grad).all()
    assert float(grad.abs().sum()) > 0.0
