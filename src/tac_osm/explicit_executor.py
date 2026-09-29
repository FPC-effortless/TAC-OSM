"""Successor explicit-graph executor.

The executable graph is now the semantic object:

    G = (V, E_true)

Candidate substrate edges remain useful only for routing/diagnostic transport.
Execution never infers hidden topology from node-local features.

The default mode is exact Boolean semantics.  The soft computation-strength
path is retained as an explicit ablation and is never used to define the exact
capability reference.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

from . import ExecutionResult, Structure
from .executor import AND, EQ, INPUT, NOT, OR, Program, XOR

__all__ = [
    "ExplicitExecutorConfig",
    "ExplicitExecutionWork",
    "ExplicitGraphExecutor",
]


@dataclass(frozen=True)
class ExplicitExecutorConfig:
    mode: str = "exact"
    max_nodes: int = 32
    alpha: tuple[float, float] = (1.0, 1.0)

    def __post_init__(self) -> None:
        if self.mode not in ("exact", "soft"):
            raise ValueError("mode must be 'exact' or 'soft'")
        if self.max_nodes < 4:
            raise ValueError("max_nodes must be >= 4")


@dataclass(frozen=True)
class ExplicitExecutionWork:
    structures_executed: int
    active_nodes: int
    active_edges: int
    structural_operations: int
    candidate_edges_metadata: int

    @staticmethod
    def for_program(program: Program) -> "ExplicitExecutionWork":
        return ExplicitExecutionWork(
            structures_executed=1,
            active_nodes=program.active_count,
            active_edges=len(program.true_edges),
            structural_operations=sum(
                1 for node in program.nodes[:program.active_count] if node.op is not INPUT
            ),
            candidate_edges_metadata=len(program.candidate_edges),
        )

    def __add__(self, other: "ExplicitExecutionWork") -> "ExplicitExecutionWork":
        if not isinstance(other, ExplicitExecutionWork):
            return NotImplemented
        return ExplicitExecutionWork(
            self.structures_executed + other.structures_executed,
            self.active_nodes + other.active_nodes,
            self.active_edges + other.active_edges,
            self.structural_operations + other.structural_operations,
            self.candidate_edges_metadata + other.candidate_edges_metadata,
        )


@dataclass(frozen=True)
class ExplicitExecution:
    result: ExecutionResult
    work: ExplicitExecutionWork


class ExplicitGraphExecutor:
    """Execute the program's declared true topology directly."""

    def __init__(self, config: ExplicitExecutorConfig | None = None) -> None:
        self.config = config if config is not None else ExplicitExecutorConfig()

    def execute(self, structure: Structure, inputs: Sequence[float]) -> ExecutionResult:
        return self.execute_with_work(structure, inputs).result

    def execute_with_work(
        self, structure: Structure, inputs: Sequence[float]
    ) -> ExplicitExecution:
        program = structure.spec
        if not isinstance(program, Program):
            raise TypeError("ExplicitGraphExecutor expects Structure.spec to be Program")

        incoming: dict[tuple[int, int], int] = {}
        for edge in program.true_edges:
            key = (edge.dst, edge.port)
            if key in incoming:
                raise ValueError(f"multiple true edges for node={edge.dst} port={edge.port}")
            incoming[key] = edge.src

        values = [0.0] * max(program.active_count, len(program.nodes))
        raw_inputs = tuple(
            float(x) for x in (program.input_values if program.input_values else inputs)
        )
        for j, node_index in enumerate(program.inputs):
            if j < len(raw_inputs):
                values[node_index] = raw_inputs[j]

        alpha = self.config.alpha
        for node in program.nodes[:program.active_count]:
            if node.op is INPUT:
                continue
            args: list[float] = []
            for port in range(node.arity):
                src = incoming.get((node.index, port))
                if src is None:
                    raise ValueError(
                        f"explicit program is incomplete: node={node.index} "
                        f"port={port} has no true edge"
                    )
                args.append(values[src])

            if self.config.mode == "exact":
                values[node.index] = self._apply_exact(node.op, args)
            else:
                values[node.index] = self._apply_soft(node.op, args, alpha)

        true_set = program.true_edge_set
        gates = tuple(
            1.0 if (edge.src, edge.dst, edge.port) in true_set else 0.0
            for edge in program.candidate_edges
        )
        result = ExecutionResult(
            output=values[program.output],
            gates=gates,
            node_values=tuple(values[: program.active_count]),
            provenance=f"explicit_graph:{self.config.mode}",
        )
        return ExplicitExecution(
            result=result,
            work=ExplicitExecutionWork.for_program(program),
        )

    @staticmethod
    def _apply_exact(op: object, args: Sequence[float]) -> float:
        a = float(args[0]) if args else 0.0
        b = float(args[1]) if len(args) > 1 else 0.0
        if op is NOT:
            return 1.0 - a
        if op is AND:
            return a * b
        if op is OR:
            return a + b - a * b
        if op is XOR:
            return abs(a - b)
        if op is EQ:
            return 1.0 - abs(a - b)
        raise ValueError(f"unsupported op in explicit executor: {op}")

    @staticmethod
    def _apply_soft(
        op: object, args: Sequence[float], alpha: Sequence[float]
    ) -> float:
        a = (float(alpha[0]) if alpha else 1.0) * (float(args[0]) if args else 0.0)
        b = (
            (float(alpha[1]) if len(alpha) > 1 else 1.0)
            * (float(args[1]) if len(args) > 1 else 0.0)
        )
        if op is NOT:
            return 1.0 - a
        if op is AND:
            return a * b
        if op is OR:
            return a + b - a * b
        if op is XOR:
            return a + b - 2.0 * a * b
        if op is EQ:
            return 1.0 - abs(a - b)
        raise ValueError(f"unsupported op in explicit executor: {op}")
