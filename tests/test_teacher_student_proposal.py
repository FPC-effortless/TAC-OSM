"""Tests for teacher-student sparse proposal routing."""

from __future__ import annotations

from tac_osm import Query
from tac_osm.c5_end_to_end import build_task, prepare_state
from tac_osm.learned_prototype_state import LearnedPrototypeStateIndex, PrototypeStateIndexConfig
from tac_osm.learned_state_index import LearnedSemanticStateIndex, LearnedStateIndexConfig
from tac_osm.teacher_student_proposal import (
    CosineTeacherStudentProposal,
    DistillationConfig,
)
from tac_osm.noisy_state_tasks import CODEBOOK


def _teacher(seed=0):
    model = LearnedSemanticStateIndex(
        LearnedStateIndexConfig(
            input_dim=10,
            latent_dim=8,
            learning_rate=0.02,
            margin=0.25,
            epochs=2,
            bucket_bits=8,
            probe_radius=1,
            shortlist_k=1,
            seed=seed,
        )
    )
    model.train_with_negative_coverage(CODEBOOK[16:32], negative_count=4, aggregation="mean")
    return model


def _proposal(seed=0):
    task = build_task(seed, 64, 0)
    state = prepare_state(task)
    teacher = _teacher(seed)
    prototypes = LearnedPrototypeStateIndex(
        teacher,
        PrototypeStateIndexConfig(prototype_count=8, bucket_capacity=8, kmeans_iterations=2),
    )
    prototypes.build(state, CODEBOOK[16:32])
    proposal = CosineTeacherStudentProposal(
        teacher,
        prototypes,
        DistillationConfig(learning_rate=0.01, epochs=2, beam_width=2, max_shortlist=16, seed=seed),
    )
    return proposal, task, state


def test_distillation_updates_query_student_without_gold_targets():
    proposal, task, _ = _proposal()
    diagnostics = proposal.fit(
        [Query(text=task.query.text, step=task.query.step, provenance="train_distill")]
    )
    assert diagnostics.optimizer_updates == 2
    assert diagnostics.teacher_candidate_score_macs > 0
    assert diagnostics.student_prototype_score_macs > 0
    assert diagnostics.final_kl >= 0.0


def test_proposal_returns_bounded_shortlist_and_cost_ledger():
    proposal, task, _ = _proposal(1)
    proposal.fit([task.query])
    lookup = proposal.lookup(task.query, beam_width=2, max_shortlist=16)
    assert len(lookup.candidate_addresses) <= 16
    assert lookup.proposal_macs == 10 * 8 + 8 * 8
    assert lookup.total_macs == lookup.proposal_macs + lookup.rerank_macs
    assert lookup.rerank_macs == 10 * 8 + len(lookup.candidate_addresses) * 8


def test_teacher_negative_filter_rejects_teacher_close_candidates():
    proposal, task, _ = _proposal(2)
    proposal.fit([task.query])
    result = proposal.filter_hard_negatives(
        task.query,
        tuple(sorted(proposal.prototypes.state_embeddings)),
        task.target_address,
        minimum_teacher_margin=0.0,
    )
    assert len(result.accepted_indices) + len(result.rejected_indices) == 63
    assert result.positive_teacher_score >= 0.0


def test_training_queries_are_not_required_to_contain_eval_targets():
    proposal, task, _ = _proposal(3)
    query = Query(
        text=" ".join(str(int(x)) for x in CODEBOOK[20]),
        step=1,
        provenance="distillation_train",
    )
    proposal.fit([query])
    assert proposal.updates == 2


def test_distillation_config_rejects_invalid_temperature():
    try:
        DistillationConfig(teacher_temperature=0.0)
    except ValueError as exc:
        assert "temperatures" in str(exc)
    else:
        raise AssertionError("non-positive temperature must fail")
