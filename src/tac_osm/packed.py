"""Packed structural graph transport for successor TAC-OSM.

This is the first packed representation, not a benchmarked GPU kernel.  It
makes active structure explicit in contiguous arrays and preserves graph
offsets, so execution work can be counted from active nodes/edges rather than
from Python container overhead.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .executor import Edge, Node, Program

__all__ = ["PackedGraphBatch"]


@dataclass(frozen=True)
class PackedGraphBatch:
    node_graph_offsets: tuple[int, ...]
    edge_graph_offsets: tuple[int, ...]
    nodes: tuple[Node, ...]
    true_edges: tuple[Edge, ...]
    outputs: tuple[int, ...]
    inputs: tuple[tuple[int, ...], ...]
    active_nodes: tuple[int, ...]
    active_edges: tuple[int, ...]
    candidate_edges: tuple[int, ...]

    @classmethod
    def from_programs(cls, programs: Sequence[Program]) -> "PackedGraphBatch":
        if not programs:
            raise ValueError("cannot pack an empty program batch")
        nodes: list[Node] = []
        true_edges: list[Edge] = []
        node_offsets: list[int] = [0]
        edge_offsets: list[int] = [0]
        outputs: list[int] = []
        inputs: list[tuple[int, ...]] = []
        active_nodes: list[int] = []
        active_edges: list[int] = []
        candidate_edges: list[int] = []

        node_base = 0
        edge_base = 0
        for program in programs:
            for node in program.nodes[:program.active_count]:
                nodes.append(
                    Node(
                        index=node.index + node_base,
                        op=node.op,
                        depth=node.depth,
                        arity=node.arity,
                    )
                )
            for edge in program.true_edges:
                true_edges.append(
                    Edge(
                        src=edge.src + node_base,
                        dst=edge.dst + node_base,
                        port=edge.port,
                    )
                )
            outputs.append(program.output + node_base)
            inputs.append(tuple(index + node_base for index in program.inputs))
            active_nodes.append(program.active_count)
            active_edges.append(len(program.true_edges))
            candidate_edges.append(len(program.candidate_edges))
            node_base += program.active_count
            edge_base += len(program.true_edges)
            node_offsets.append(node_base)
            edge_offsets.append(edge_base)

        return cls(
            node_graph_offsets=tuple(node_offsets),
            edge_graph_offsets=tuple(edge_offsets),
            nodes=tuple(nodes),
            true_edges=tuple(true_edges),
            outputs=tuple(outputs),
            inputs=tuple(inputs),
            active_nodes=tuple(active_nodes),
            active_edges=tuple(active_edges),
            candidate_edges=tuple(candidate_edges),
        )

    @property
    def n_graphs(self) -> int:
        return len(self.active_nodes)

    @property
    def total_active_nodes(self) -> int:
        return sum(self.active_nodes)

    @property
    def total_active_edges(self) -> int:
        return sum(self.active_edges)

    @property
    def total_candidate_edges(self) -> int:
        return sum(self.candidate_edges)

    def validate(self) -> None:
        if self.node_graph_offsets[-1] != len(self.nodes):
            raise AssertionError("node offsets do not match packed node array")
        if self.edge_graph_offsets[-1] != len(self.true_edges):
            raise AssertionError("edge offsets do not match packed edge array")
        for graph, (ns, ne) in enumerate(
            zip(self.node_graph_offsets, self.node_graph_offsets[1:])
        ):
            if ne - ns != self.active_nodes[graph]:
                raise AssertionError("active node span mismatch")
        for graph, (es, ee) in enumerate(
            zip(self.edge_graph_offsets, self.edge_graph_offsets[1:])
        ):
            if ee - es != self.active_edges[graph]:
                raise AssertionError("active edge span mismatch")
