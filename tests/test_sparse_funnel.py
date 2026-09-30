"""Tests for the integrated sparse retrieval funnel."""

from __future__ import annotations

from tac_osm import Query
from tac_osm.c5_end_to_end import build_task, prepare_state
from tac_osm.learned_prototype_state import LearnedPrototypeStateIndex, PrototypeStateIndexConfig
from tac_osm.learned_state_index import LearnedSemanticStateIndex, LearnedStateIndexConfig
from tac_osm.sparse_funnel import SparseRetrievalFunnel
from tac_osm.teacher_student_proposal import CosineTeacherStudentProposal, DistillationConfig
from tac_osm.noisy_state_tasks import CODEBOOK


def _stack(seed=0):
    teacher = LearnedSemanticStateIndex(LearnedStateIndexConfig(
        input_dim=10, latent_dim=8, learning_rate=0.02, margin=0.25, epochs=2,
        bucket_bits=8, probe_radius=1, shortlist_k=1, seed=seed,
    ))
    teacher.train_with_negative_coverage(CODEBOOK[16:32], negative_count=4)
    task = build_task(seed, 64, 0)
    state = prepare_state(task)
    prototypes = LearnedPrototypeStateIndex(
        teacher, PrototypeStateIndexConfig(prototype_count=8, bucket_capacity=8, kmeans_iterations=2)
    )
    prototypes.build(state, CODEBOOK[16:32])
    proposal = CosineTeacherStudentProposal(
        teacher, prototypes, DistillationConfig(epochs=1, beam_width=2, max_shortlist=16, seed=seed)
    )
    proposal.fit([Query(text=task.query.text, step=0, provenance="distillation_train")])
    return proposal, task, state


def test_funnel_builds_packed_layout_from_teacher_state_embeddings():
    proposal, task, state = _stack()
    del task
    funnel = SparseRetrievalFunnel(proposal)
    layout = funnel.build_packed_layout()
    assert layout.state_items == len(state.addresses()) == 64
    assert layout.embedding_dim == 8


def test_funnel_lookup_preserves_bounded_candidate_set_and_rows():
    proposal, task, _ = _stack(1)
    funnel = SparseRetrievalFunnel(proposal)
    result = funnel.lookup(task.query, beam_width=2, max_shortlist=16)
    assert len(result.proposal.candidate_addresses) <= 16
    assert len(result.contiguous_rows) == len(result.proposal.candidate_addresses)
    assert result.layout["candidate_rows"] == len(result.proposal.candidate_addresses)
