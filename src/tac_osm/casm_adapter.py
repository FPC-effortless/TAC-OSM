"""Adapter boundary for a real CASM structural executor.

The hardened default executor is a synthetic relation control. This module
makes the replacement point explicit without importing CASM into TAC-OSM's
minimal dependency surface.

The adapter accepts a callable supplied by the CASM implementation. TAC-OSM
only translates the narrow Structure/inputs contract into ExecutionResult.
No CASM performance or correctness claim is made here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence

from . import ExecutionResult, StructuralExecutor, Structure


CasmExecuteFn = Callable[[Any, Sequence[float]], ExecutionResult | Any]


@dataclass(frozen=True)
class CasmExecutorAdapter:
    """Thin dependency-injection adapter for an external CASM executor."""

    execute_fn: CasmExecuteFn

    def __post_init__(self) -> None:
        if not callable(self.execute_fn):
            raise TypeError("execute_fn must be callable")

    def execute(
        self,
        structure: Structure,
        inputs: Sequence[float],
    ) -> ExecutionResult:
        result = self.execute_fn(structure.spec, inputs)
        if isinstance(result, ExecutionResult):
            return result
        if isinstance(result, (int, float)):
            value = float(result)
            return ExecutionResult(
                output=value,
                node_values=(value,),
                provenance="casm_adapter.scalar",
            )
        if isinstance(result, dict):
            return ExecutionResult(
                output=float(result["output"]),
                gates=tuple(float(x) for x in result.get("gates", ())),
                node_values=tuple(float(x) for x in result.get("node_values", ())),
                provenance=str(result.get("provenance", "casm_adapter")),
            )
        raise TypeError(
            "CASM execute_fn must return ExecutionResult, numeric output, or "
            "a mapping containing 'output'"
        )

    def __call__(self, structure: Structure, inputs: Sequence[float]) -> ExecutionResult:
        return self.execute(structure, inputs)
