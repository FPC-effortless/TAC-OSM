#!/usr/bin/env python3
"""Cheap identifiability diagnostic for the pinned CASM-S executor."""
from __future__ import annotations
import itertools, os, sys
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]
CASM_SOURCE_ROOT = Path(os.environ.get("CASM_SOURCE_ROOT", str(ROOT / "third_party" / "cdl-attention-experiment")))
if not CASM_SOURCE_ROOT.exists():
    raise RuntimeError(f"CASM source root does not exist: {CASM_SOURCE_ROOT}")
sys.path.insert(0, str(CASM_SOURCE_ROOT))

from casm_v01.phase1_dag.grammar import Edge, Node, Op
from casm_v01.phase1_dag.generator import Episode
from casm_v01.phase1_dag.model import CASMS

def make_episode(parent0: int, parent1: int) -> Episode:
    nodes = [
        Node(0, Op.INPUT, 0, 0), Node(1, Op.INPUT, 0, 1),
        Node(2, Op.INPUT, 0, 2), Node(3, Op.INPUT, 0, 3),
        Node(4, Op.AND, 1, 4), Node(5, Op.NOT, 2, 5),
    ]
    true_edges = (Edge(parent0, 4, 0), Edge(parent1, 4, 1), Edge(4, 5, 0))
    candidate_edges = tuple(
        Edge(src, dst, port)
        for dst in range(4, 6)
        for port in range(nodes[dst].arity)
        for src in range(dst)
    )
    table = {}
    for bits in itertools.product((0, 1), repeat=4):
        table[bits] = 1 - int(bits[parent0] & bits[parent1])
    return Episode(nodes, true_edges, candidate_edges, (0,1,2,3), 5,
                   (0,0,0,0), table[(0,0,0,0)], table, 6)

def main() -> None:
    left, right = make_episode(0, 1), make_episode(2, 3)
    if all(left.truth_table[k] == right.truth_table[k] for k in left.truth_table):
        raise AssertionError("diagnostic construction failed")
    model = CASMS(max_nodes=10, dim=32, temperature=2.0, seed=0).eval()
    with torch.no_grad():
        left_h, _ = model.structural_encode([left])
        right_h, _ = model.structural_encode([right])
        left_gates, _ = model.gate([left])
        right_gates, _ = model.gate([right])
    if not torch.equal(left_h, right_h):
        raise AssertionError("public structural encodings differ unexpectedly")
    if not torch.equal(left_gates, right_gates):
        raise AssertionError("CASM-S gate differs despite identical public descriptors")
    print("M2 executor identifiability diagnostic: PASS")
    print("public_structural_encoding_equal=true")
    print("gate_values_equal=true")
    print("truth_tables_equal=false")
    print("conclusion=hidden wiring is not identifiable by the registered CASM-S gate")

if __name__ == "__main__":
    main()
