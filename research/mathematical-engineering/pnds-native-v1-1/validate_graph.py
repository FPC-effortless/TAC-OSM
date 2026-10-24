#!/usr/bin/env python3
"""Fail-closed S0 native source-DAG governance; NOT a whole-model equivalence proof."""
import json
import copy
from pathlib import Path

PATH = Path(__file__).with_name("WHOLE_GRAPH_CONTRACT.json")
EXPECTED = (
    "encoded", "addresses", "contents", "old_memory_read_t",
    "fast_t", "slow_t", "write_gates_t", "memory_t",
    "final_read", "codebook", "class_logits",
)
ROOTS = {"kind", "keys", "values", "memory_prev", "fast_prev", "slow_prev", "parameters"}


def verify(data):
    if data.get("schema") != "tac_osm.native_operator_dag.v1.1":
        raise ValueError("source typed DAG schema changed")
    source = data.get("source", {})
    if (source.get("commit") != "081e534a074001c6a3b2251b3ff31ad20ec4fca8"
        or source.get("path") != "src/tac_osm/aligned_associative_tac.py"):
        raise ValueError("source-pinned native forward changed without versioning")
    nodes = data.get("nodes", [])
    if tuple(n["id"] for n in nodes) != EXPECTED:
        raise ValueError("missing/altered input-to-output source DAG node")
    seen = set(ROOTS)
    for n in nodes:
        if not n.get("shape") or not n.get("op") or not n.get("parallel"):
            raise ValueError(f"node without tensor/math/parallel semantics: {n['id']}")
        if not n.get("dependencies") or set(n["dependencies"]) - seen:
            raise ValueError(f"invalid future/cyclic dependency: {n['id']}")
        seen.add(n["id"])
    if (set(data.get("inputs",{})) != {"kind","keys","values"}
        or "causal history loop" not in data["state"]["temporal_semantics"]
        or "read_only" not in data["state"]["observed_roles"]):
        raise ValueError("time/role/input contract lost")
    gates = data["certification"]
    if (gates["legally_scan_parallel_whole_graph"] is not False
        or gates["native_full_graph_semantic_Learn_proof"] != "OPEN"
        or gates["native_full_graph_PyTorch_forward_backward_reference"] != "OPEN"
        or gates["old_key_long_history_retention"] != "OPEN"
        or gates["independent_real_skill"] != "NONE_ESTABLISHED"):
        raise ValueError("unproven whole native model promoted")
    return {"nodes":len(nodes), "pinned_source":True,
            "acyclic_declared_dependency_graph":True,
            "full_model_scan_or_skill_proven":False}


def selftest(data):
    bad = copy.deepcopy(data)
    bad["nodes"][3]["dependencies"] = ["class_logits"]
    try: verify(bad)
    except ValueError: pass
    else: raise AssertionError("future output leak not rejected")
    bad = copy.deepcopy(data)
    bad["certification"]["native_full_graph_semantic_Learn_proof"] = "PASS"
    try: verify(bad)
    except ValueError: pass
    else: raise AssertionError("false proof claim not rejected")
    bad = copy.deepcopy(data)
    bad["source"]["commit"] = "0"*40
    try: verify(bad)
    except ValueError: pass
    else: raise AssertionError("source drift not rejected")


if __name__ == "__main__":
    data = json.loads(PATH.read_text())
    selftest(data)
    print(json.dumps(verify(data),sort_keys=True))
