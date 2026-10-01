"""Structural tests for TACOSM-C5-SPARSE-FUNNEL-001."""

from __future__ import annotations

from tac_osm import Query
from tac_osm.c5_end_to_end import build_task, prepare_state
from tac_osm.learned_prototype_state import LearnedPrototypeStateIndex, PrototypeStateIndexConfig
from tac_osm.learned_state_index import LearnedSemanticStateIndex, LearnedStateIndexConfig
from tac_osm.noisy_state_tasks import CODEBOOK
from tac_osm.teacher_student_proposal import CosineTeacherStudentProposal, DistillationConfig


def _stack(seed=0):
    teacher = LearnedSemanticStateIndex(LearnedStateIndexConfig(
        input_dim=10, latent_dim=8, learning_rate=0.02, margin=0.25, epochs=2,
        bucket_bits=8, probe_radius=1, shortlist_k=1, seed=seed,
    ))
    teacher.train_with_negative_coverage(CODEBOOK[16:32], negative_count=4, aggregation="mean")
    task = build_task(seed, 64, 0)
    state = prepare_state(task)
    prototypes = LearnedPrototypeStateIndex(
        teacher, PrototypeStateIndexConfig(prototype_count=8, bucket_capacity=8, kmeans_iterations=2),
    )
    prototypes.build(state, CODEBOOK[16:32])
    proposal = CosineTeacherStudentProposal(
        teacher, prototypes,
        DistillationConfig(epochs=1, beam_width=2, max_shortlist=16, seed=seed),
    )
    proposal.fit([Query(text=" ".join(str(int(x)) for x in CODEBOOK[20]), step=0, provenance="distillation_train")])
    return proposal, task


def test_k_budget_maps_to_bounded_prototype_beam():
    proposal, task = _stack()
    for k, beam in ((4, 1), (8, 2), (16, 4)):
        lookup = proposal.lookup(task.query, beam_width=beam, max_shortlist=k)
        assert len(lookup.candidate_addresses) <= k


def test_teacher_distribution_is_full_state_distribution():
    proposal, task = _stack(1)
    addresses, distribution = proposal.teacher_distribution(task.query)
    assert len(addresses) == 64
    assert len(distribution) == 64
    assert abs(sum(distribution) - 1.0) < 1e-9


def test_distillation_is_target_free():
    proposal, task = _stack(2)
    query = Query(text=" ".join(str(int(x)) for x in CODEBOOK[21]), step=0, provenance="distillation_train")
    diagnostics = proposal.fit([query])
    assert diagnostics.optimizer_updates == 1
    assert query.context == ()
    assert query.provenance == "distillation_train"


def test_teacher_negative_filter_partitions_candidates_when_capped():
    proposal, task = _stack(3)
    addresses = tuple(sorted(proposal.prototypes.state_embeddings))
    result = proposal.filter_hard_negatives(
        task.query, addresses, task.target_address, minimum_teacher_margin=0.0, max_negatives=8
    )
    assert len(result.accepted_indices) == 8
    assert len(result.accepted_indices) + len(result.rejected_indices) == 63


def test_lookup_cost_has_separate_student_and_teacher_terms():
    proposal, task = _stack(4)
    lookup = proposal.lookup(task.query, beam_width=2, max_shortlist=16)
    expected = 10 * 8 + 8 * 8 + 10 * 8 + len(lookup.candidate_addresses) * 8
    assert lookup.total_macs == expected
