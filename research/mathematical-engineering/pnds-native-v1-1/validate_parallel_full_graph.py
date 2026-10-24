#!/usr/bin/env python3
"""Fail-closed schema of entire optional time-parallel TAC input→output graph.

This checks advertised topological/causal decomposition, not source
equivalence, mathematical theorem validity or measured real-world skills.
"""
import json
from pathlib import Path

PATH=Path(__file__).with_name("PARALLEL_FULL_GRAPH_CONTRACT.json")
EXPECTED=(
"input","embeddings","fast_basis","prefix_context","provisional_fast",
"provisional_slow","input_gates","norm_budget_gate","affine_pairs",
"associative_prefix","readonly_exact_trace","old_read","refined_fast",
"refined_slow","final_read","readout_logits",
)


def verify(data):
    if data.get("schema")!="tac_osm.native_parallel_candidate.input_output_dag.v1.4":
        raise ValueError("wrong complete graph schema")
    if data["source"]["pinned_commit"]!="081e534a074001c6a3b2251b3ff31ad20ec4fca8":
        raise ValueError("unversioned source drift")
    candidate=data["candidate"]
    if (candidate["parameters_added"]!=0 or
        candidate["trainable_parameter_count"]!=101527 or
        candidate["source_outputs_equivalent"] is not False or
        candidate["whole_time_sequence_parallel_algorithms"] is not True or
        candidate["all_17_Tac_variants_engineered"] is not False or
        candidate["real_world_capability_claim"]!="NONE"):
        raise ValueError("unearned source/skill/portfolio or budget claim")
    ns=data["nodes"]
    names=tuple(node["id"] for node in ns)
    if names!=EXPECTED or len(set(names))!=len(names):
        raise ValueError("complete pipeline node missing, reordered or duplicated")
    seen=set()
    for n in ns:
        if not (n["op"] and n["shape"] and n["parallel"] and n["causal"]):
            raise ValueError("untyped or noncausal computation node")
        if set(n["deps"])-seen:
            raise ValueError("future-loop or nonexistent dependency at "+n["id"])
        seen.add(n["id"])
    gates=names.index("input_gates")
    refine=names.index("refined_fast")
    if gates>=refine or "refined_fast" in next(
           n for n in ns if n["id"]=="input_gates")["deps"]:
        raise ValueError("state-dependent gate cycle invalidates scan")
    if next(n for n in ns if n["id"]=="associative_prefix")["deps"]!=["affine_pairs"]:
        raise ValueError("memory schedule may no longer be independent")
    if next(n for n in ns if n["id"]=="old_read")["deps"] != [
            "readonly_exact_trace","embeddings"]:
        raise ValueError("old read must depend only on previous prefix and key")
    if (data["verification"]["training_steps"]!=0 or
        data["verification"]["real_skill"]!="NOT_ESTABLISHED"):
        raise ValueError("model capability falsely promoted")
    return {"complete_nodes":len(ns),"strict_causal_topological_order":True,
            "parallel_memory_schedule_is_upstream_of_refinement":True,
            "source_equivalence_not_claimed":True,"real_skill_claim":False}


def adversarial_tests(data):
    import copy
    tests=[]
    bad=copy.deepcopy(data)
    bad["nodes"][6]["deps"].append("refined_fast")
    tests.append(bad)
    bad=copy.deepcopy(data)
    bad["candidate"]["source_outputs_equivalent"]=True
    tests.append(bad)
    bad=copy.deepcopy(data)
    bad["verification"]["real_skill"]="PROVEN"
    tests.append(bad)
    bad=copy.deepcopy(data)
    bad["nodes"].pop()
    tests.append(bad)
    for i,bad in enumerate(tests):
        try: verify(bad)
        except ValueError: pass
        else: raise AssertionError(f"dangerous graph mutation {i} not rejected")


if __name__=="__main__":
    data=json.loads(PATH.read_text())
    adversarial_tests(data)
    print(json.dumps(verify(data),sort_keys=True))
