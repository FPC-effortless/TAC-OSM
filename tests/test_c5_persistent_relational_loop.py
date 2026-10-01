"""Tests for C5 persistent relational full loop."""
from tac_osm.c5_persistent_relational_loop import (
    DIM,
    N_OPS,
    apply_relation,
    build_population,
    corrupted_op_context,
    encode_state_value,
    decode_state_value,
    make_episode,
    persistent_oracle_rank,
)
from tac_osm.temporal import TemporalPersistentState


def test_relation_is_not_identity_to_either_operand() -> None:
    left = (0,) * DIM
    right = (1,) * DIM
    assert apply_relation(left, right, 0) == (1,) * DIM
    assert apply_relation(left, right, 1) == (0,) * DIM
    assert apply_relation(left, right, 2) == (0,) * DIM
    assert apply_relation(left, right, 3) == (1,) * DIM


def test_persistent_state_stores_operands_not_target() -> None:
    left = tuple(i % 2 for i in range(DIM))
    right = tuple((i + 1) % 2 for i in range(DIM))
    op = 0
    encoded = encode_state_value(left, right, op)
    decoded_left, decoded_right, decoded_op = decode_state_value(encoded)
    assert decoded_left == left
    assert decoded_right == right
    assert decoded_op == op
    assert len(encoded) == 2 * DIM + 1
    assert apply_relation(left, right, op) not in (left, right)


def test_corrupted_query_hint_differs_from_clean_one_hot() -> None:
    for op in range(N_OPS):
        hint = corrupted_op_context(op, seed=op + 1)
        assert len(hint) == N_OPS
        assert hint != tuple(int(i == op) for i in range(N_OPS))


def test_temporal_persistence_survives_three_boundaries_and_interference() -> None:
    trial, state = make_episode(seed=0, step=0, m=64)
    assert state.current_step == 3
    read = state.read(trial.query)
    assert read.values
    left, right, op = decode_state_value(read.values[0])
    assert apply_relation(left, right, op) == trial.target_descriptor
    assert trial.target_descriptor not in (
        tuple(read.values[0][:DIM]),
        tuple(read.values[0][DIM:2 * DIM]),
    )
    assert len(state.addresses()) >= 2


def test_reset_removes_persistent_relation() -> None:
    trial, state = make_episode(seed=1, step=0, m=64)
    assert state.read(trial.query).values
    state.clear()
    assert state.read(trial.query).values == ()


def test_population_contains_exactly_one_target() -> None:
    trial, _ = make_episode(seed=2, step=1, m=128)
    matches = [i for i, c in enumerate(trial.candidates) if c.descriptor == trial.target_descriptor]
    assert matches == [trial.target_index]


def test_oracle_upper_bound_is_rank_one() -> None:
    trial, _ = make_episode(seed=3, step=2, m=64)
    assert persistent_oracle_rank(trial) == 1
