from tac_osm.fused_architecture import OperatorDescriptor
from tac_osm.operator_learning import PrimitiveOperator, TransitionRecord, StructMeans, apply_operator, PSTLearner
from tac_osm.structural_product_key import StructuralProductKeyRouter


def _records():
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
    return rows


def test_type_weighted_structural_rerank_prefers_exact_kind():
    candidates = [
        __import__("tac_osm", fromlist=["Candidate"]).Candidate(
            key="toggle",
            descriptor=OperatorDescriptor("toggle", (1,0,1,0,1,0,1,0)).encode(),
            action=0,
        ),
        __import__("tac_osm", fromlist=["Candidate"]).Candidate(
            key="set1",
            descriptor=OperatorDescriptor("set1", (1,0,1,0,1,0,1,0)).encode(),
            action=1,
        ),
    ]
    router = StructuralProductKeyRouter(factor_size=2, factor_beam=2)
    q = tuple(map(float, OperatorDescriptor("toggle", (1,0,1,0,1,0,1,0)).encode()))
    assert router._exact_score(candidates[0], (q,)) > router._exact_score(candidates[1], (q,))


def test_factor_geometry_5x8_is_constructible():
    rows = _records()
    candidates = [
        __import__("tac_osm", fromlist=["Candidate"]).Candidate(
            key=f"c{i}",
            descriptor=OperatorDescriptor(r.operator.kind, r.operator.mask).encode(),
            action=i,
        )
        for i, r in enumerate(rows)
    ]
    router = StructuralProductKeyRouter(factor_count=5, factor_size=8, factor_beam=4)
    router.build(candidates, training_records=rows)
    assert router.index.build_diagnostics.factor_count == 5
