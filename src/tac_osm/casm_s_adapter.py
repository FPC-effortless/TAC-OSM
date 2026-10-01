"""Pinned external CASM-S adapter for TACOSM-M2-CASM-SELECTIVE-009.

The TAC-OSM core remains dependency-free. This adapter is imported only by the
research runner after the exact external CASM-S commit has been checked out.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class WorkAccounting:
    gate_evaluations: int
    edge_message_operations: int
    structural_node_operations: int

    @property
    def total(self) -> int:
        return (
            self.gate_evaluations
            + self.edge_message_operations
            + self.structural_node_operations
        )


class CASMSAdapter:
    """Thin adapter around casm_v01.phase1_dag.model.CASMS."""

    def __init__(self, casm_model, torch_module):
        self.model = casm_model
        self.torch = torch_module
        self.device = next(casm_model.parameters()).device

    @staticmethod
    def work_for_episode(episode) -> WorkAccounting:
        input_nodes = sum(1 for n in episode.nodes[:episode.active_count] if n.op.value == "INPUT")
        structural_nodes = max(0, episode.active_count - input_nodes)
        edge_count = len(episode.candidate_edges)
        return WorkAccounting(
            gate_evaluations=edge_count,
            edge_message_operations=edge_count,
            structural_node_operations=structural_nodes,
        )

    def encode_structure(self, episodes: Sequence[object]):
        """Frozen node-local CASM-S representation; true edges are not consumed."""
        self.model.eval()
        with self.torch.no_grad():
            h, _ = self.model.structural_encode(list(episodes))
            return h.mean(dim=1)

    def execute_batch(self, episodes: Sequence[object], runtime_inputs):
        """Execute a batch and return outputs without exposing oracle topology."""
        self.model.eval()
        with self.torch.no_grad():
            x = self.torch.tensor(runtime_inputs, dtype=self.torch.float32, device=self.device)
            outputs, gates, _meta = self.model(list(episodes), x)
        return outputs.detach().cpu().tolist()
