"""Auditable dense arithmetic omitted by CPU fused attention profiler entries.

Multiply-add=2 convention. Only attention matrix products are supplemented;
normalization, exp/softmax, masking, kernel execution and optimizer work are
not magically converted into a complete hardware FLOP measurement.
"""

FORWARD = "aten::_scaled_dot_product_flash_attention_for_cpu"
BACKWARD = FORWARD + "_backward"


def dense_attention_units(query, key, value, *, backward=False):
    if any(len(s) != 4 or any(type(v) is not int or v <= 0 for v in s)
           for s in (query, key, value)):
        raise ValueError("Need observed positive [batch, heads, time, dimension] shapes")
    b, h, tq, dq = query
    bk, hk, tk, dk = key
    bv, hv, tv, dv = value
    if (b, h) != (bk, hk) or (b, h) != (bv, hv) or tk != tv or dq != dk:
        raise ValueError("Observed attention batch/head/key/value shapes disagree")
    # Forward: QK^T and PV; backward: dQ,dK,dP,dV.
    return (4 if backward else 2)*b*h*tq*tk*(dq+dv)


def supplement_record(record):
    if record["operator"] not in (FORWARD, BACKWARD):
        raise ValueError("Unregistered fused attention operator")
    if record["already_profiled_flops"] or record["dense_matmul_children"]:
        raise ValueError("Would double-charge an already counted dense operation")
    names = record["schema_argument_names"]
    shapes = record["observed_input_shapes"]
    def shape(name):
        index=names.index(name)
        if index >= len(shapes): raise ValueError("Missing runtime argument shape")
        return shapes[index]
    return dense_attention_units(shape("query"),shape("key"),shape("value"),
                                  backward=record["operator"] == BACKWARD)


def from_profiler(torch, profiler):
    rows=[]
    for event in profiler.events():
        if event.name not in (FORWARD, BACKWARD): continue
        op=event.name.removeprefix("aten::")
        schema=getattr(torch.ops.aten,op).default._schema
        children=[]
        def walk(e):
            for child in e.cpu_children:
                if child.name in ("aten::mm","aten::bmm","aten::addmm","aten::baddbmm"):
                    children.append(child.name)
                walk(child)
        walk(event)
        record={"operator":event.name,"schema":str(schema),
            "schema_argument_names":[a.name for a in schema.arguments],
            "observed_input_shapes":event.input_shapes,
            "already_profiled_flops":event.flops or 0,"dense_matmul_children":children}
        record["supplemental_matrix_product_units"]=supplement_record(record)
        rows.append(record)
    return {"convention":"scalar multiply-add=2; full dense-equivalent QK/PV and four backward products, not measured instructions",
            "events":rows,"supplemental_dense_attention_units":sum(r["supplemental_matrix_product_units"] for r in rows),
            "complete_training_flops":False,
            "exclusions":["attention score scaling","softmax/exp/normalization/backward elementwise work",
                "causal masking and memory operations; a fused causal kernel may skip future blocks",
                "optimizer and runtime kernel implementation/instruction counts"]}
