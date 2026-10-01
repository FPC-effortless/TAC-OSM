from tac_osm.operator_learning import (
    AXONConsolidator,
    ExperienceStore,
    FixedAXONConsolidator,
    PrimitiveOperator,
    PSTLearner,
    SECAEngine,
    StructMeans,
    TransitionRecord,
    apply_operator,
)


def _records(n=30, dim=8):
    from random import Random
    rng = Random(123)
    kinds = ("toggle", "set1", "set0")
    rows = []
    for i in range(n):
        state = tuple(rng.randrange(2) for _ in range(dim))
        kind = kinds[i % len(kinds)]
        mask = tuple(1 if ((i + j) % 3 == 0) else 0 for j in range(dim))
        op = PrimitiveOperator(kind, mask)
        rows.append(
            TransitionRecord(
                before=state,
                operator=op,
                after=apply_operator(state, op),
                verified=True,
                episode=i,
                step=0,
            )
        )
    return rows


def test_pst_generalizes_transition_law_to_unseen_mask():
    rows = _records(60)
    pst = PSTLearner(("toggle", "set1", "set0"))
    pst.fit(rows)
    test = rows[7]
    unseen_mask = tuple(1 if i % 2 else 0 for i in range(8))
    op = PrimitiveOperator(test.operator.kind, unseen_mask)
    assert pst.predict(test.before, op) == apply_operator(test.before, op)


def test_structmeans_finds_reusable_structure():
    rows = _records(90)
    sm = StructMeans(3, ("toggle", "set1", "set0"), seed=3)
    sm.fit(rows)
    assert len(sm.centroids) == 3
    assert sm.purity(rows) >= 0.90
    assert sm.compression_ratio(rows) == 30.0


def test_axon_produces_parameterized_reusable_operators():
    rows = _records(30)
    axon = AXONConsolidator(min_support=4)
    macros = axon.consolidate(rows)
    assert {m.source_kind for m in macros} == {"toggle", "set1", "set0"}
    goal = apply_operator(rows[0].before, PrimitiveOperator("toggle", (1, 0, 1, 0, 1, 0, 1, 0)))
    toggle_macro = next(m for m in macros if m.source_kind == "toggle")
    bound = toggle_macro.bind_to_goal(rows[0].before, goal)
    assert bound.execute(rows[0].before) == goal


def test_regm_rebuilds_pst_and_rejects_unverified():
    rows = _records(12)
    store = ExperienceStore()
    for row in rows:
        assert store.append(row)
    bad = TransitionRecord(
        before=rows[0].before,
        operator=rows[0].operator,
        after=tuple(1 - x for x in rows[0].after),
        verified=False,
        episode=99,
        step=0,
    )
    assert not store.append(bad)
    pst = store.reconstruct_pst(("toggle", "set1", "set0"))
    assert pst.transition_accuracy(rows) == 1.0


def test_seca_accepts_novel_compositions():
    rows = _records(60)
    pst = PSTLearner(("toggle", "set1", "set0"))
    pst.fit(rows)
    macros = FixedAXONConsolidator(min_support=4).consolidate(rows)
    candidates = SECAEngine().propose(macros, max_pairs=4)
    accepted = SECAEngine().verify(
        candidates,
        [rows[i].before for i in range(10)],
        lambda state, macro: macro.execute(state),
    )
    assert candidates
    assert accepted
    assert all(op.name.startswith("seca:") for op in accepted)
