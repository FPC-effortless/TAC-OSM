from __future__ import annotations
import random

from tac_osm.plm_unified import (
    AdaptiveCASM, BinaryCDL, HierarchicalStateIndex, MemoryRecord,
    OperatorPlan, PLMConfig, StateKind, StateStatus, TypedPersistentState,
    Verification, build_operator_pool, generate_episode, new_plm_for_records,
)


def record(i: int, bits: tuple[int, ...], op: str = "xor") -> MemoryRecord:
    return MemoryRecord(
        f"r{i}", "relational", bits, (i & 1, (i // 2) & 1),
        StateKind.WORLD, StateStatus.VERIFIED, i, "test", 1.0, None, op,
    )


def test_verified_commit_gates_state():
    s = TypedPersistentState()
    p = s.propose(record_id="x", namespace="n", key_bits=(1,0),
                  payload=(1,0), state_kind=StateKind.EXPERIENCE, step=1)
    assert not s.commit(p, Verification(False, 0, "reject"), step=1)
    assert s.size == 0
    assert s.commit(p, Verification(True, 1, "accept"), step=1)
    assert s.get("x").status is StateStatus.VERIFIED


def test_index_reports_raw_bucket_and_cap_separately():
    rs = [record(i, tuple((i >> j) & 1 for j in range(8))) for i in range(64)]
    idx = HierarchicalStateIndex((2,4,6))
    idx.build(rs)
    ids, probes, raw, fallback = idx.lookup(rs[10].key_bits, max_candidates=4)
    assert "r10" in ids
    assert raw >= len(ids)
    assert probes <= 8
    assert not fallback


def test_router_learns_delayed_relevance():
    cfg = PLMConfig(slow_lr=0.05, fast_lr=0.10)
    router = BinaryCDL(8, config=cfg)
    rs = [record(i, tuple((i >> j) & 1 for j in range(8))) for i in range(32)]
    target = rs[17]
    for _ in range(60):
        ranked = router.rank(target.key_bits, rs)
        selected = next(r for r in rs if r.record_id == ranked[0].record_id)
        router.update(target.key_bits, selected,
                      1.0 if selected.record_id == target.record_id else -1.0,
                      correct_record=target)
    assert router.rank(target.key_bits, rs)[0].record_id == target.record_id


def test_casm_halting_is_explicit_and_bounded():
    pool = build_operator_pool()
    pool.synthesize_not()
    r = record(0, (0,) * 8)
    early = AdaptiveCASM(pool, PLMConfig(max_depth=3, halt_threshold=0.8)).execute(
        r, OperatorPlan(("not",), confidence=1.0, estimated_cost=1)
    )
    deep = AdaptiveCASM(pool, PLMConfig(max_depth=3, halt_threshold=0.8)).execute(
        r, OperatorPlan(("not", "not", "not"), confidence=0.2, estimated_cost=3)
    )
    assert early.halted_at == 1
    assert deep.halted_at == 3


def test_false_acceptance_stress_can_commit_wrong_experience():
    task, rs = generate_episode(rng=random.Random(4), history=4, records_target=5, noise=0.0)
    cfg = PLMConfig(false_accept_rate=1.0, max_admitted=4, max_executions=1)
    plm = new_plm_for_records(rs, config=cfg)
    result = plm.step(task, step=1)
    assert result.verification.passed
    assert result.verification.false_accept
    assert plm.state.writes >= 1


def test_privileged_online_correction_is_opt_in():
    assert PLMConfig().allow_privileged_online_correction is False
    assert PLMConfig(allow_privileged_online_correction=True).allow_privileged_online_correction is True


def test_composite_operator_rejects_unsupported_binary_second_stage():
    pool = build_operator_pool()
    pool.synthesize_not()
    pool.synthesize_composite("not_xor", "xor", "not")
    assert "not_xor" in pool.operators
    import pytest
    with pytest.raises(ValueError, match="unary second operators"):
        pool.synthesize_composite("bad", "xor", "and")
