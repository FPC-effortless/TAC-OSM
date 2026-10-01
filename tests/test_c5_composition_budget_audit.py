"""Tests for C5 composition-representation and budget audit."""
from tac_osm.c5_composition_budget_audit import (
    ProductKeyRelationRouter,
    BitwiseLateInteractionRouter,
    AnalyticRelationalRouter,
    analytic_relational_initialize,
    exact_relation_hash_lookup,
    verify_split_disjointness,
)
from tac_osm.c5_persistent_relational_loop import (
    DIM,
    apply_relation,
    make_episode,
    prepare_state,
    CDLPersistentRelationRouter,
)

def test_analytic_initialization_matches_boolean_relation_basis() -> None:
    router = AnalyticRelationalRouter(seed=0, learning_rate=0.012, soft_target_epsilon=0.05)
    analytic_relational_initialize(router)
    for op in range(4):
        left = tuple((i + op) % 2 for i in range(DIM))
        right = tuple((i * 3 + op) % 2 for i in range(DIM))
        trial, _ = make_episode(seed=10 + op, step=op, m=64)
        state, query = prepare_state(seed=10 + op, step=op, address=trial.address, left=left, right=right, op=op)
        qz = router.encode_query(query, state)
        expected = apply_relation(left, right, op)
        assert tuple(1 if x >= 0 else 0 for x in qz) == expected

def test_product_key_router_has_two_factorized_subspaces() -> None:
    router = ProductKeyRelationRouter(seed=0)
    assert len(router.keys0) == 256
    assert len(router.keys1) == 256
    assert len(router.keys0[0]) == 8
    trial, _ = make_episode(seed=1, step=0, m=64)
    assert len(router.encode_candidate(trial.candidates[0])) == 16

def test_late_interaction_router_has_one_channel_per_bit() -> None:
    router = BitwiseLateInteractionRouter(seed=0)
    trial, state = make_episode(seed=2, step=0, m=64)
    assert len(router.encode_query(trial.query, state)) == DIM
    assert len(router.encode_candidate(trial.candidates[0])) == DIM

def test_exact_relation_hash_reference_is_rank_one_and_four_lookups() -> None:
    trial, state = make_episode(seed=3, step=0, m=64)
    rank, lookups = exact_relation_hash_lookup(trial, state)
    assert rank == 1
    assert lookups == 4

def test_calibration_and_heldout_streams_are_disjoint() -> None:
    check = verify_split_disjointness(seed=4, m=1024)
    assert check["calibration_count"] == 64
    assert check["heldout_count"] == 64
    assert check["intersection_count"] == 0
    assert check["calibration_digest"] != check["heldout_digest"]
