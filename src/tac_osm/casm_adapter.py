"""Torch-free boundary for integrating the CASM-S structural executor.

The CASM-S implementation in cdl-attention-experiment is a torch module and
therefore stays outside this repository's control-plane runtime. This module
freezes the data contract between the two repositories:

    CASM-S Episode -> CasmGraphSpec -> Structure.spec -> CasmExecuteFn
                                             \
                                              -> ExecutionResult

Only public graph structure crosses the boundary. In particular, the adapter
never serializes true_edges, target, input_values or truth_table. Runtime
inputs are passed separately to the executor. Candidate-edge order is
preserved exactly because CASM-S indexes its gate tensor by the position of an
edge in episode.candidate_edges.

No torch import belongs here. The injected execute_fn may use torch on the
compute runner; this module must remain importable on Termux.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Callable

from . import ExecutionResult, Structure


CASM_SCHEMA = "casm-s/1"
CASM_KIND = "structural_graph"


def _op_name(op: Any) -> str:
    """Return the stable string value used by CASM-S grammar.Op."""
    value = getattr(op, "value", None)
    if value is not None:
        return str(value)
    name = getattr(op, "name", None)
    if name is not None:
        return str(name)
    return str(op)


def _require_int(name: str, value: Any, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an int")
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return value


@dataclass(frozen=True)
class CasmNodeSpec:
    """Portable representation of the CASM-S Node contract.

    slot is retained even though the current CASM-S structural encoder derives
    positional information from index; retaining it avoids throwing away a
    public node field at the repository boundary.
    """

    index: int
    op: str
    depth: int
    slot: int
    arity: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "op": self.op,
            "depth": self.depth,
            "slot": self.slot,
            "arity": self.arity,
        }


@dataclass(frozen=True)
class CasmEdgeSpec:
    """Portable representation of one CASM-S candidate edge.

    index is the position in Episode.candidate_edges. It is part of the wire
    format so gate vectors can be checked against the exact ordering produced
    by CASM-S rather than relying on an implicit reordering.
    """

    index: int
    src: int
    dst: int
    port: int

    def to_dict(self) -> dict[str, int]:
        return {
            "index": self.index,
            "src": self.src,
            "dst": self.dst,
            "port": self.port,
        }


@dataclass(frozen=True)
class CasmGraphSpec:
    """Torch-free, JSON-serializable structural graph contract.

    The schema is intentionally not an executable oracle description. It
    contains the candidate graph and its addressing metadata, but never the
    episode's hidden truth fields.
    """

    nodes: tuple[CasmNodeSpec, ...]
    candidate_edges: tuple[CasmEdgeSpec, ...]
    inputs: tuple[int, ...]
    output: int
    active_count: int

    def __post_init__(self) -> None:
        if not self.nodes:
            raise ValueError("CASM graph must contain at least one node")
        _require_int("active_count", self.active_count)
        if self.active_count > len(self.nodes):
            raise ValueError("active_count cannot exceed node count")

        node_indices = [node.index for node in self.nodes]
        if len(set(node_indices)) != len(node_indices):
            raise ValueError("node indices must be unique")
        if any(i < 0 or i >= len(self.nodes) for i in node_indices):
            raise ValueError("node index must be within the serialized node range")

        for input_index in self.inputs:
            _require_int("input index", input_index)
            if input_index >= self.active_count:
                raise ValueError("input node must be active")

        _require_int("output", self.output)
        if self.output >= self.active_count:
            raise ValueError("output node must be active")

        edge_indices = [edge.index for edge in self.candidate_edges]
        if edge_indices != list(range(len(self.candidate_edges))):
            raise ValueError(
                "candidate-edge indices must be contiguous and preserve list order"
            )

        for edge in self.candidate_edges:
            if edge.src < 0 or edge.src >= self.active_count:
                raise ValueError("candidate edge source must be active")
            if edge.dst < 0 or edge.dst >= self.active_count:
                raise ValueError("candidate edge destination must be active")
            if edge.src >= edge.dst:
                raise ValueError("CASM-S candidate edges must be topologically ordered")
            dst_node = self.nodes[edge.dst]
            if edge.port < 0 or edge.port >= dst_node.arity:
                raise ValueError(
                    f"edge {edge.index} has invalid port {edge.port} for "
                    f"destination arity {dst_node.arity}"
                )

    def to_dict(self) -> dict[str, Any]:
        """Return a plain mapping suitable for Structure.spec."""
        return {
            "schema": CASM_SCHEMA,
            "kind": CASM_KIND,
            "active_count": self.active_count,
            "nodes": [node.to_dict() for node in self.nodes],
            "candidate_edges": [edge.to_dict() for edge in self.candidate_edges],
            "inputs": list(self.inputs),
            "output": self.output,
        }

    @classmethod
    def from_episode(cls, episode: Any) -> "CasmGraphSpec":
        """Translate a real CASM-S Episode without importing CASM-S.

        The translation relies only on the public field names used by the
        source dataclasses: nodes, candidate_edges, inputs, output and
        active_count.
        """
        nodes = tuple(
            CasmNodeSpec(
                index=_require_int("node.index", node.index),
                op=_op_name(node.op),
                depth=_require_int("node.depth", node.depth),
                slot=_require_int("node.slot", node.slot),
                arity=_require_int("node.arity", node.arity),
            )
            for node in episode.nodes
        )
        candidate_edges = tuple(
            CasmEdgeSpec(
                index=index,
                src=_require_int("edge.src", edge.src),
                dst=_require_int("edge.dst", edge.dst),
                port=_require_int("edge.port", edge.port),
            )
            for index, edge in enumerate(episode.candidate_edges)
        )
        inputs = tuple(
            _require_int("episode input", index) for index in episode.inputs
        )
        return cls(
            nodes=nodes,
            candidate_edges=candidate_edges,
            inputs=inputs,
            output=_require_int("episode.output", episode.output),
            active_count=_require_int("episode.active_count", episode.active_count),
        )


def casm_graph_spec_from_episode(episode: Any) -> dict[str, Any]:
    """Serialize a CASM-S episode to the portable Structure.spec mapping."""
    return CasmGraphSpec.from_episode(episode).to_dict()


def casm_structure_from_episode(
    episode: Any,
    *,
    key: str,
    provenance: str = "casm_s",
) -> Structure:
    """Build a TAC-OSM Structure from a CASM-S Episode."""
    if not key:
        raise ValueError("key must be non-empty")
    return Structure(
        key=key,
        spec=casm_graph_spec_from_episode(episode),
        provenance=provenance,
    )


CasmExecuteFn = Callable[[Mapping[str, Any], Sequence[float]], ExecutionResult | Any]


def _scalar_float(name: str, value: Any) -> float:
    """Convert scalar/tensor-like values without importing a tensor library."""
    item = getattr(value, "item", None)
    if callable(item):
        value = item()
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be scalar-convertible") from exc


def _flat_float_tuple(name: str, value: Any) -> tuple[float, ...]:
    """Convert a one-dimensional tensor/list/tuple-like value to Python floats."""
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        value = tolist()

    if isinstance(value, (str, bytes)) or isinstance(value, Mapping):
        raise TypeError(f"{name} must be a one-dimensional numeric sequence")

    try:
        values = tuple(value)
    except TypeError as exc:
        raise TypeError(f"{name} must be a one-dimensional numeric sequence") from exc

    if any(isinstance(item, (list, tuple, Mapping)) for item in values):
        raise ValueError(f"{name} must be one-dimensional")
    return tuple(_scalar_float(name, item) for item in values)


def _validate_result_lengths(
    spec: CasmGraphSpec,
    *,
    gates: tuple[float, ...],
    node_values: tuple[float, ...],
) -> None:
    expected_gates = len(spec.candidate_edges)
    if len(gates) != expected_gates:
        raise ValueError(
            f"CASM-S gates length {len(gates)} != candidate-edge count "
            f"{expected_gates}"
        )
    if len(node_values) != spec.active_count:
        raise ValueError(
            f"CASM-S node_values length {len(node_values)} != active_count "
            f"{spec.active_count}"
        )


@dataclass(frozen=True)
class CasmExecutorAdapter:
    """Dependency-injection adapter for the external CASM-S executor."""

    execute_fn: CasmExecuteFn

    def __post_init__(self) -> None:
        if not callable(self.execute_fn):
            raise TypeError("execute_fn must be callable")

    def execute(
        self,
        structure: Structure,
        inputs: Sequence[float],
    ) -> ExecutionResult:
        spec_obj = structure.spec
        typed_spec = spec_obj if isinstance(spec_obj, CasmGraphSpec) else None
        spec = typed_spec.to_dict() if typed_spec is not None else spec_obj

        expected: CasmGraphSpec | None = typed_spec
        if expected is None and isinstance(spec, Mapping) and spec.get("schema") == CASM_SCHEMA:
            expected = _parse_graph_spec(spec)

        if expected is not None and len(inputs) != len(expected.inputs):
            raise ValueError(
                f"CASM-S inputs length {len(inputs)} != graph input count "
                f"{len(expected.inputs)}"
            )

        result = self.execute_fn(spec, inputs)

        if isinstance(result, ExecutionResult):
            normalized = result
        elif isinstance(result, (int, float)):
            value = _scalar_float("output", result)
            normalized = ExecutionResult(
                output=value,
                node_values=(value,),
                provenance="casm_adapter.scalar",
            )
        elif isinstance(result, Mapping):
            if "output" not in result:
                raise TypeError("CASM execute_fn mapping result must contain 'output'")
            normalized = ExecutionResult(
                output=_scalar_float("output", result["output"]),
                gates=_flat_float_tuple("gates", result.get("gates", ())),
                node_values=_flat_float_tuple(
                    "node_values", result.get("node_values", ())
                ),
                provenance=str(result.get("provenance", "casm_adapter")),
            )
        else:
            raise TypeError(
                "CASM execute_fn must return ExecutionResult, numeric output, or "
                "a mapping containing 'output'"
            )

        if expected is not None:
            _validate_result_lengths(
                expected,
                gates=normalized.gates,
                node_values=normalized.node_values,
            )

        return normalized

    def __call__(self, structure: Structure, inputs: Sequence[float]) -> ExecutionResult:
        return self.execute(structure, inputs)


def _parse_graph_spec(spec: Mapping[str, Any]) -> CasmGraphSpec:
    """Re-validate a serialized CASM graph before handing it to the executor."""
    try:
        nodes = tuple(
            CasmNodeSpec(
                index=_require_int("node.index", node["index"]),
                op=str(node["op"]),
                depth=_require_int("node.depth", node["depth"]),
                slot=_require_int("node.slot", node["slot"]),
                arity=_require_int("node.arity", node["arity"]),
            )
            for node in spec["nodes"]
        )
        edges = tuple(
            CasmEdgeSpec(
                index=_require_int("edge.index", edge["index"]),
                src=_require_int("edge.src", edge["src"]),
                dst=_require_int("edge.dst", edge["dst"]),
                port=_require_int("edge.port", edge["port"]),
            )
            for edge in spec["candidate_edges"]
        )
        inputs = tuple(_require_int("input index", index) for index in spec["inputs"])
        return CasmGraphSpec(
            nodes=nodes,
            candidate_edges=edges,
            inputs=inputs,
            output=_require_int("output", spec["output"]),
            active_count=_require_int("active_count", spec["active_count"]),
        )
    except KeyError as exc:
        raise TypeError(f"invalid CASM-S graph spec; missing {exc.args[0]!r}") from exc


__all__ = [
    "CASM_SCHEMA",
    "CASM_KIND",
    "CasmNodeSpec",
    "CasmEdgeSpec",
    "CasmGraphSpec",
    "CasmExecuteFn",
    "CasmExecutorAdapter",
    "casm_graph_spec_from_episode",
    "casm_structure_from_episode",
]
