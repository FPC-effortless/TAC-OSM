from tac_osm import Candidate
from tac_osm.fused_architecture import (
    OperatorDescriptor,
    build_structmeans_and_pst,
    candidate_operator,
    candidate_record,
    structural_query_scores,
)
from tac_osm.operator_learning import PrimitiveOperator, TransitionRecord, apply_operator


def test_operator_descriptor_round_trip():
    desc = OperatorDescriptor("set1", (1, 0, 1, 0, 0, 1, 0, 1))
    assert OperatorDescriptor.decode(desc.encode()) == desc


def test_candidate_record_is_verified_structural_witness():
    candidate = Candidate(
        key="x",
        descriptor=OperatorDescriptor("toggle", (1, 0, 1, 0, 1, 0, 0, 1)).encode(),
        action=0,
    )
    rec = candidate_record(candidate)
    assert rec.verified
    assert rec.operator == PrimitiveOperator("toggle", (1, 0, 1, 0, 1, 0, 0, 1))
    assert rec.after == apply_operator(rec.before, rec.operator)


def test_structmeans_and_pst_share_verified_experience():
    rows = []
    for kind in ("toggle", "set1", "set0"):
        for mask in (
            (1,0,1,0,1,0,1,0),
            (0,1,0,1,0,1,0,1),
            (1,1,0,0,1,1,0,0),
        ):
            op = PrimitiveOperator(kind, mask)
            state = (0,1,0,1,1,0,0,1)
            rows.append(TransitionRecord(
                before=state,
                operator=op,
                after=apply_operator(state, op),
                verified=True,
                episode=0,
                step=0,
            ))
    sm, pst = build_structmeans_and_pst(rows, clusters=3)
    assert sm.purity(rows) >= 0.90
    assert pst.transition_accuracy(rows) == 1.0
    scores = structural_query_scores(sm, rows[0].before, rows[0].after)
    assert len(scores) == 3
