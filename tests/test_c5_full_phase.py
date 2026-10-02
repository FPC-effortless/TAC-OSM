"""Unit/gate tests for TACOSM-C5-FULL-PHASE-001."""
from __future__ import annotations

from tac_osm import Candidate, Query, StateUpdate
from tac_osm.c5_full_phase import (
    CASMVerifier,
    CDLDenseOutcomeRouter,
    ORLSHIndex,
    conformal_k,
)
from tac_osm.temporal import TemporalPersistentState


def test_casm_exact_relation_has_one_success() -> None:
    casm = CASMVerifier(dim=4, max_nodes=15)
    reference = (1, 0, 1, 0)
    candidates = (
        Candidate("good", reference, 0),
        Candidate("bad", (1, 1, 1, 0), 1),
        Candidate("bad2", (0, 0, 1, 0), 2),
    )
    outcomes = [
        casm.execute_and_verify(candidate=c, reference=reference)
        for c in candidates
    ]
    assert [o.verified and o.output >= 0.5 for o in outcomes] == [True, False, False]


def test_temporal_write_is_not_visible_before_boundary() -> None:
    state = TemporalPersistentState()
    key = "world:test"
    value = (1, 0, 1, 0)
    state.stage_world_write(StateUpdate(key=key, value=value, step=0), delay=1)
    query = Query(text="\t" + key, context=(1, 1, 1, 1), step=0)
    assert state.read(query).values == ()
    state.advance_to(1)
    assert state.read(query).values == (value,)


def test_conformal_k_is_bounded_and_monotone_with_coverage() -> None:
    ranks = [1, 1, 2, 2, 3, 4, 5, 6, 8, 10]
    k90 = conformal_k(ranks, 0.10, 32)
    k95 = conformal_k(ranks, 0.05, 32)
    assert 1 <= k90 <= 32
    assert k90 <= k95


def test_lsh_lookup_uses_only_admitted_candidates() -> None:
    router = CDLDenseOutcomeRouter(
        dim=4,
        latent_dim=4,
        seed=0,
        learning_rate=0.01,
        soft_target_epsilon=0.05,
        analytic_init=True,
    )
    candidates = tuple(
        Candidate(f"c{i}", tuple((i >> j) & 1 for j in range(4)), i)
        for i in range(16)
    )
    embeddings = router.candidate_embeddings(candidates)
    index = ORLSHIndex(dim=4, bits=4, tables=8, seed=11, tables_cap=128)
    index.build(embeddings)
    state = TemporalPersistentState()
    address = "world:q"
    state.stage_world_write(StateUpdate(address, (1, 0, 0, 0), 0), delay=1)
    state.advance_to(1)
    query = Query(text="\t" + address, context=(1, 1, 1, 1), step=1)
    zq = router.encode_query(query, state)
    lookup = index.lookup(
        zq,
        candidate_limit=4,
        score_fn=lambda a: router.score_embeddings(zq, index.embeddings[a]),
    )
    assert len(lookup.addresses) <= 4
    assert lookup.candidate_rerank_count >= len(lookup.addresses)


def test_verifier_feedback_can_update_router() -> None:
    router = CDLDenseOutcomeRouter(
        dim=4,
        latent_dim=4,
        seed=0,
        learning_rate=0.01,
        soft_target_epsilon=0.05,
    )
    candidates = (
        Candidate("good", (1, 0, 1, 0), 0),
        Candidate("bad", (1, 1, 1, 0), 1),
    )
    state = TemporalPersistentState()
    key = "world:feedback"
    state.stage_world_write(StateUpdate(key, (1, 0, 1, 0), 0), delay=1)
    state.advance_to(1)
    query = Query(text="\t" + key, context=(1, 1, 1, 1), step=1)
    before = router.updates
    router.learn_from_verified(query, state, candidates, 0, 1)
    assert router.updates == before + 1
