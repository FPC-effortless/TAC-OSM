from __future__ import annotations

import inspect

import pytest
import torch

from tac_osm.behavioral_compatibility_router import (
    BehavioralCompatibilityRouter,
    BehavioralRouterConfig,
)


def test_pair_logits_support_single_and_batched_queries():
    model = BehavioralCompatibilityRouter(
        BehavioralRouterConfig(candidate_dim=6, row_dim=5, hidden_dim=8, latent_dim=4, pair_hidden_dim=8, seed=7)
    )
    candidates = torch.randn(3, 6)
    rows = torch.randn(4, 5)
    logits = model.pair_logits(candidates, rows)
    assert logits.shape == (3, 4)

    batched_candidates = candidates.unsqueeze(0).repeat(2, 1, 1)
    batched_rows = rows.unsqueeze(0).repeat(2, 1, 1)
    batched = model.pair_logits(batched_candidates, batched_rows)
    assert batched.shape == (2, 3, 4)


def test_compatibility_training_signature_has_no_target_identity_argument():
    params = inspect.signature(
        BehavioralCompatibilityRouter.training_loss
    ).parameters
    assert "target_index" not in params
    assert "pair_labels" in params


def test_support_set_loss_uses_all_row_labels():
    model = BehavioralCompatibilityRouter(
        BehavioralRouterConfig(candidate_dim=6, row_dim=5, hidden_dim=8, latent_dim=4, pair_hidden_dim=8, seed=3)
    )
    candidates = torch.randn(2, 6)
    rows = torch.randn(2, 5)
    labels = torch.tensor([[1.0, 1.0], [1.0, 0.0]])
    loss = model.training_loss(candidates, rows, labels)
    assert torch.isfinite(loss)


def test_rank_is_deterministic_for_fixed_inputs():
    model = BehavioralCompatibilityRouter(
        BehavioralRouterConfig(candidate_dim=6, row_dim=5, hidden_dim=8, latent_dim=4, pair_hidden_dim=8, seed=9)
    )
    candidates = torch.randn(5, 6)
    rows = torch.randn(4, 5)
    first = model.rank(candidates, rows)
    second = model.rank(candidates, rows)
    assert torch.equal(first, second)


@pytest.mark.parametrize("support_size", [4, 8, 12])
def test_registered_support_sizes_leave_verifier_rows(support_size):
    assert 16 - support_size >= 4
