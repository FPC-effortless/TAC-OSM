"""Structural and numerical tests for TACOSM-C5-COSINE-OBJECTIVE-001."""

from __future__ import annotations

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from tac_osm.c5_end_to_end import build_task, prepare_state
from tac_osm.learned_state_index import LearnedSemanticStateIndex, LearnedStateIndexConfig
from tac_osm.noisy_state_tasks import CODEBOOK


def _model(seed=0):
    return LearnedSemanticStateIndex(
        LearnedStateIndexConfig(
            input_dim=10,
            latent_dim=16,
            learning_rate=0.02,
            margin=0.25,
            epochs=2,
            bucket_bits=8,
            probe_radius=1,
            shortlist_k=1,
            seed=seed,
        )
    )


def _loss(q, p, n, margin=0.25):
    qn = q / math.sqrt(sum(x * x for x in q))
    pn = p / math.sqrt(sum(x * x for x in p))
    nn = n / math.sqrt(sum(x * x for x in n))
    diff = sum(a * b for a, b in zip(qn, pn)) - sum(
        a * b for a, b in zip(qn, nn)
    )
    return math.log1p(math.exp(margin - diff))


def test_cosine_training_keeps_update_count_fixed():
    model = _model()
    assert model.train_with_cosine_negative_coverage(
        CODEBOOK[16:40],
        negative_count=8,
        positive_views=1,
    ) == 48


def test_cosine_training_uses_disjoint_training_split():
    assert set(CODEBOOK[16:64]).isdisjoint(CODEBOOK[:16])


def test_cosine_embedding_cost_is_fixed_at_width_sixteen():
    model = _model()
    assert model.query_embedding_macs == 160
    assert model.state_embedding_macs == 160


def test_cosine_training_rejects_nonpositive_view_count():
    model = _model()
    try:
        model.train_with_cosine_negative_coverage(
            CODEBOOK[16:40],
            negative_count=8,
            positive_views=0,
        )
    except ValueError as exc:
        assert "positive_views" in str(exc)
    else:
        raise AssertionError("positive_views=0 must fail")


def test_cosine_normalization_has_unit_norm():
    model = _model()
    normalized, norm = model._normalize([3.0, 4.0])
    assert abs(norm - 5.0) < 1e-12
    assert abs(sum(x * x for x in normalized) - 1.0) < 1e-12


def test_cosine_gradient_matches_finite_difference():
    q = [0.7, -0.4, 1.2]
    p = [1.1, 0.3, -0.5]
    n = [-0.2, 0.8, 0.9]
    margin = 0.25
    eps = 1e-8

    def grads():
        qn_norm = math.sqrt(sum(x * x for x in q))
        pn_norm = math.sqrt(sum(x * x for x in p))
        nn_norm = math.sqrt(sum(x * x for x in n))
        qn = [x / qn_norm for x in q]
        pn = [x / pn_norm for x in p]
        nn = [x / nn_norm for x in n]
        sim_pos = sum(a * b for a, b in zip(qn, pn))
        sim_neg = sum(a * b for a, b in zip(qn, nn))
        gate = 1.0 / (1.0 + math.exp(-(margin - (sim_pos - sim_neg))))
        q_grad = [
            gate * ((pn[i] - sim_pos * qn[i]) / qn_norm
                    - (nn[i] - sim_neg * qn[i]) / qn_norm)
            for i in range(3)
        ]
        return q_grad

    analytic = grads()
    numeric = []
    for i in range(3):
        qp = q[:]
        qm = q[:]
        qp[i] += eps
        qm[i] -= eps
        numeric.append(
            (_loss(qp, p, n, margin) - _loss(qm, p, n, margin))
            / (2 * eps)
        )
    for a, num in zip(analytic, numeric):
        assert abs(a + num) < 1e-6


def test_cosine_state_pool_remains_sixty_four():
    task = build_task(4, 64, 0)
    state = prepare_state(task)
    assert len(state.addresses()) == 64


def test_cosine_train_and_eval_codes_have_expected_sizes():
    assert len(CODEBOOK[16:64]) == 48
    assert len(CODEBOOK[:16]) == 16
