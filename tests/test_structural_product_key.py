from tac_osm.fused_architecture import OperatorDescriptor
from tac_osm.operator_learning import PrimitiveOperator, TransitionRecord, StructMeans, apply_operator
from tac_osm.structural_product_key import StructuralProductKeyRouter, _descriptor_vector


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


def test_structural_descriptor_is_compositional():
    desc = OperatorDescriptor("toggle", (1,0,1,0,1,0,1,0))
    assert _descriptor_vector(desc.kind, desc.mask) == tuple(map(float, desc.encode()))


def test_structural_product_key_builds_only_from_verified_training_signatures():
    rows = _records()
    candidates = []
    for idx, row in enumerate(rows):
        candidates.append(
            __import__("tac_osm", fromlist=["Candidate"]).Candidate(
                key=f"c{idx}",
                descriptor=OperatorDescriptor(row.operator.kind, row.operator.mask).encode(),
                action=idx,
            )
        )
    router = StructuralProductKeyRouter(factor_size=8)
    router.build(candidates, training_records=rows)
    assert router.index.build_diagnostics.codebook_training_items == 9
    assert len(router.index.build_diagnostics.factor_dims) == 3
