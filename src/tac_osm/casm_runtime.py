"""Optional compute-runner bridge for the external CASM-S implementation.

This module is intentionally outside TAC-OSM dependency-light core. It imports
torch and the external CASM repository only when CasmSRuntime is constructed
on a compute runner.

The source checkout is pinned to CASM-S commit:
c31554413301e3c9d3e6b3f8c8c6be572a74a748

The bridge reconstructs a CASM-S Episode from the torch-free graph transported
by tac_osm.casm_adapter. Hidden true wiring is not reconstructed: the
CASM-S forward path does not need it, and supplying it here would turn a
runtime transport into an oracle channel.

The runner exposes explicit work units so C5 can distinguish candidate-count
reduction from mere routing-call reduction.
"""

from __future__ import annotations

import importlib
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from . import ExecutionResult, Structure
from .casm_adapter import CasmGraphSpec, _parse_graph_spec

CASM_REPO = "FPC-effortless/cdl-attention-experiment"
CASM_COMMIT = "c31554413301e3c9d3e6b3f8c8c6be572a74a748"


@dataclass(frozen=True)
class CasmExecutionWork:
    """Mechanical work accounting for one CASM-S graph execution."""

    structures_executed: int
    active_nodes: int
    candidate_edges: int
    gate_evaluations: int
    executed_structural_operations: int
    node_outputs: int

    def __add__(self, other: "CasmExecutionWork") -> "CasmExecutionWork":
        if not isinstance(other, CasmExecutionWork):
            return NotImplemented
        return CasmExecutionWork(
            structures_executed=self.structures_executed + other.structures_executed,
            active_nodes=self.active_nodes + other.active_nodes,
            candidate_edges=self.candidate_edges + other.candidate_edges,
            gate_evaluations=self.gate_evaluations + other.gate_evaluations,
            executed_structural_operations=(
                self.executed_structural_operations + other.executed_structural_operations
            ),
            node_outputs=self.node_outputs + other.node_outputs,
        )

    @staticmethod
    def zero() -> "CasmExecutionWork":
        return CasmExecutionWork(0, 0, 0, 0, 0, 0)

    @staticmethod
    def for_spec(spec: CasmGraphSpec) -> "CasmExecutionWork":
        non_input_ops = sum(
            1 for node in spec.nodes[:spec.active_count] if node.arity > 0
        )
        edges = len(spec.candidate_edges)
        return CasmExecutionWork(
            structures_executed=1,
            active_nodes=spec.active_count,
            candidate_edges=edges,
            gate_evaluations=edges,
            executed_structural_operations=non_input_ops,
            node_outputs=spec.active_count,
        )


@dataclass(frozen=True)
class CasmExecution:
    """One normalized execution result plus its mechanical work units."""

    result: ExecutionResult
    work: CasmExecutionWork


@dataclass
class _RuntimeEpisode:
    nodes: tuple[Any, ...]
    candidate_edges: tuple[Any, ...]
    inputs: tuple[int, ...]
    output: int
    active_count: int

    @property
    def true_edge_set(self) -> frozenset[tuple[int, int, int]]:
        return frozenset()


class CasmSRuntime:
    """Load a pinned CASM-S model and execute TAC-OSM graph specs."""

    def __init__(
        self,
        *,
        casm_root: str | Path,
        checkpoint: str | Path | None = None,
        max_nodes: int = 10,
        dim: int = 32,
        temperature: float = 2.0,
        seed: int = 0,
        device: str | None = None,
        require_commit: str = CASM_COMMIT,
    ) -> None:
        self.casm_root = Path(casm_root).expanduser().resolve()
        if not self.casm_root.exists():
            raise FileNotFoundError(f"CASM repository not found: {self.casm_root}")
        self._assert_commit(require_commit)

        import sys
        root_string = str(self.casm_root)
        if root_string not in sys.path:
            sys.path.insert(0, root_string)

        torch = importlib.import_module("torch")
        model_module = importlib.import_module("casm_v01.phase1_dag.model")
        grammar_module = importlib.import_module("casm_v01.phase1_dag.grammar")

        self.torch = torch
        self.grammar = grammar_module
        self.device = torch.device(
            device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model = model_module.CASMS(
            max_nodes=max_nodes, dim=dim, temperature=temperature, seed=seed
        ).to(self.device)
        self.max_nodes = max_nodes
        self.dim = dim
        self.temperature = temperature
        self.seed = seed
        self.checkpoint = None
        if checkpoint is not None:
            self.checkpoint = Path(checkpoint).expanduser().resolve()
            self.load_checkpoint(self.checkpoint)
        self.model.eval()

    def _assert_commit(self, expected: str) -> None:
        result = subprocess.run(
            ["git", "-C", str(self.casm_root), "rev-parse", "HEAD"],
            check=True, capture_output=True, text=True
        )
        actual = result.stdout.strip()
        if actual != expected:
            raise RuntimeError(f"CASM-S source mismatch: expected {expected}, found {actual}")

    def load_checkpoint(self, path: Path) -> None:
        if not path.is_file():
            raise FileNotFoundError(f"CASM-S checkpoint not found: {path}")
        payload = self.torch.load(path, map_location=self.device, weights_only=False)
        state = payload
        if isinstance(payload, dict):
            for key in ("state_dict", "model_state_dict", "casm_state_dict"):
                if key in payload:
                    state = payload[key]
                    break
        if not isinstance(state, dict):
            raise TypeError("CASM-S checkpoint does not contain a state dict")
        self.model.load_state_dict(state, strict=True)
        self.model.eval()

    def save_checkpoint(self, path: str | Path, *, metadata: dict[str, Any] | None = None) -> Path:
        destination = Path(path).expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        self.torch.save(
            {
                "state_dict": self.model.state_dict(),
                "casm_commit": CASM_COMMIT,
                "max_nodes": self.max_nodes,
                "dim": self.dim,
                "temperature": self.temperature,
                "metadata": metadata or {},
            },
            destination,
        )
        return destination

    def _episode_from_spec(self, spec: CasmGraphSpec) -> _RuntimeEpisode:
        op_by_name = {str(op.value): op for op in self.grammar.Op}
        nodes = tuple(
            self.grammar.Node(
                index=node.index, op=op_by_name[node.op],
                depth=node.depth, slot=node.slot
            )
            for node in spec.nodes
        )
        edges = tuple(
            self.grammar.Edge(src=edge.src, dst=edge.dst, port=edge.port)
            for edge in spec.candidate_edges
        )
        return _RuntimeEpisode(
            nodes=nodes, candidate_edges=edges, inputs=spec.inputs,
            output=spec.output, active_count=spec.active_count
        )

    def _spec(self, structure: Structure) -> CasmGraphSpec:
        raw = structure.spec
        if isinstance(raw, CasmGraphSpec):
            return raw
        if not isinstance(raw, dict):
            raise TypeError("CasmSRuntime expects a CASM graph mapping or CasmGraphSpec")
        if raw.get("schema") != "casm-s/1":
            raise ValueError("unsupported CASM graph schema")
        return _parse_graph_spec(raw)

    def _sync(self) -> None:
        if self.device.type == "cuda":
            self.torch.cuda.synchronize(self.device)

    def execute_many(
        self, structures: Sequence[Structure], input_rows: Sequence[Sequence[float]]
    ) -> tuple[tuple[CasmExecution, ...], CasmExecutionWork]:
        if not structures:
            raise ValueError("cannot execute an empty structure set")
        if len(structures) != len(input_rows):
            raise ValueError(f"structures/input_rows mismatch: {len(structures)} != {len(input_rows)}")

        specs = tuple(self._spec(structure) for structure in structures)
        episodes = tuple(self._episode_from_spec(spec) for spec in specs)
        max_inputs = max(len(spec.inputs) for spec in specs)
        rows = []
        for spec, row in zip(specs, input_rows):
            if len(row) != len(spec.inputs):
                raise ValueError(f"input row length {len(row)} != graph input count {len(spec.inputs)}")
            rows.append([float(x) for x in row] + [0.0] * (max_inputs - len(row)))
        runtime_inputs = self.torch.tensor(rows, dtype=self.torch.float32, device=self.device)
        self._sync()
        with self.torch.no_grad():
            outputs, gates, _meta = self.model(list(episodes), runtime_inputs)
            node_matrix = self.model.last_node_values
        self._sync()

        all_work = CasmExecutionWork.zero()
        executions = []
        for b, spec in enumerate(specs):
            edge_count = len(spec.candidate_edges)
            active_count = spec.active_count
            result = ExecutionResult(
                output=float(outputs[b].item()),
                gates=tuple(float(x) for x in gates[b, :edge_count].detach().cpu().tolist()),
                node_values=tuple(float(x) for x in node_matrix[b, :active_count].detach().cpu().tolist()),
                provenance="casm_s",
            )
            work = CasmExecutionWork.for_spec(spec)
            executions.append(CasmExecution(result=result, work=work))
            all_work = all_work + work
        return tuple(executions), all_work

    def execute(self, structure: Structure, inputs: Sequence[float]) -> CasmExecution:
        executions, _work = self.execute_many((structure,), (inputs,))
        return executions[0]

__all__ = ["CASM_REPO", "CASM_COMMIT", "CasmExecutionWork", "CasmExecution", "CasmSRuntime"]