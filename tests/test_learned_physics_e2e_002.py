import json

import torch

from tac_osm.learned_physics_e2e_002 import (
    LearnedPhysicsE2E002,
    LearnedPhysicsE2E002Config,
    hamiltonian_prior_loss,
)
from tac_osm.learned_physics_e2e_benchmark import (
    ACTION_COUNT,
    HISTORY,
    benchmark_manifest,
    sample_history_pairs,
)


def test_history_pair_has_identical_current_multimodal_surface():
    import random

    left, right = sample_history_pairs(random.Random(8), 1)[0]
    for first, second in zip(left["observations"][-1], right["observations"][-1]):
        assert torch.equal(first, second)
    assert left["action"] != right["action"]


def test_model_has_no_task_specific_physical_ontology():
    model = LearnedPhysicsE2E002(LearnedPhysicsE2E002Config())
    assert sum(parameter.numel() for parameter in model.parameters()) > 0
    assert not hasattr(model, "entity_ids")
    assert not hasattr(model, "operator_table")
    assert not hasattr(model, "fixed_executor")
    assert not hasattr(model, "particle_count")
    assert not any(
        isinstance(module, (torch.nn.AdaptiveAvgPool1d, torch.nn.AdaptiveAvgPool2d))
        for module in model.modules()
    )


def test_hamiltonian_prior_is_on_shared_learned_latent():
    model = LearnedPhysicsE2E002(LearnedPhysicsE2E002Config())
    text = torch.randint(0, 32, (2, HISTORY, 5))
    image = torch.rand(2, HISTORY, 1, 32, 32)
    audio = torch.rand(2, HISTORY, 1, 96)
    goal = torch.rand(2, 2)
    logits, canonical, energy = model(text, image, audio, goal)
    assert logits.shape == (2, ACTION_COUNT)
    assert canonical.shape == (2, HISTORY, 64)
    assert energy.shape == (2, HISTORY)
    loss = hamiltonian_prior_loss(canonical, energy, 0.08)["total"]
    gradient = torch.autograd.grad(loss, canonical, retain_graph=True)[0]
    assert torch.isfinite(gradient).all()


def test_action_path_consumes_canonical_latent():
    from tac_osm.learned_physics_e2e_002 import LearnedActionHead

    head = LearnedActionHead(64, ACTION_COUNT)
    state = torch.randn(2, 64, requires_grad=True)
    canonical = torch.randn(2, 64, requires_grad=True)
    goal = torch.randn(2, 2)
    logits = head(state, canonical, goal)
    gradient = torch.autograd.grad(logits.sum(), canonical)[0]
    assert torch.isfinite(gradient).all()
    assert float(gradient.abs().sum()) > 0.0


def test_physics_manifest_matches_registered_contract():
    with open(
        "contracts/TACOSM-PLM-LEARNED-PHYSICS-E2E-002.json",
        encoding="utf-8",
    ) as handle:
        contract = json.load(handle)
    manifest = benchmark_manifest()
    assert contract["protocol"]["benchmark_generator_version"] == manifest["generator_version"]
    assert contract["physics_prior"]["allowed_information"] == manifest["physics_prior"]
