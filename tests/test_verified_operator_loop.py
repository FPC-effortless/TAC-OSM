from tac_osm.operator_learning import (
    PrimitiveOperator,
    TransitionRecord,
    VerifiedOperatorLoop,
    apply_operator,
)


def test_verified_operator_loop_ingests_rebuilds_and_discovers():
    records = []
    ops = (
        PrimitiveOperator("toggle", (0, 1, 0, 1, 0, 1, 0, 1)),
        PrimitiveOperator("set1", (1, 0, 1, 0, 1, 0, 1, 0)),
        PrimitiveOperator("set0", (1, 1, 0, 0, 1, 1, 0, 0)),
    )
    for episode, op in enumerate(ops):
        for step in range(20):
            state = tuple((episode + step + j) % 2 for j in range(8))
            records.append(
                TransitionRecord(
                    before=state,
                    operator=op,
                    after=apply_operator(state, op),
                    verified=True,
                    episode=episode,
                    step=step,
                )
            )

    loop = VerifiedOperatorLoop(
        ("toggle", "set1", "set0"),
        structmeans_k=3,
        axon_min_support=4,
    )
    update = loop.ingest_many(records)
    assert update == len(records)
    assert loop.pst.transition_accuracy(records) == 1.0
    assert len(loop.macros) == 3
    assert loop.structmeans.signature_purity(records) >= 0.85

    states = [r.before for r in records[:8]]
    novel = loop.discover(
        states,
        lambda state, macro: macro.execute(state),
        max_pairs=6,
    )
    assert len(novel) >= 1
    assert len(loop.library) > len(loop.macros)
