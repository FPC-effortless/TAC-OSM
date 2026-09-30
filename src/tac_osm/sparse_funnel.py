"""Integrated sparse persistent-state retrieval funnel.

Composes the learned student proposal, full cosine teacher reranker and the
logical packed memory layout. CASM/exact execution stays downstream and is
intentionally not imported here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from . import Query
from .packed_state import PackedStateLayout
from .teacher_student_proposal import CosineTeacherStudentProposal, DistilledProposalLookup


@dataclass(frozen=True)
class SparseFunnelLookup:
    proposal: DistilledProposalLookup
    layout: dict[str, int | float]
    selected_row: int | None
    addressing_mode: Literal["prototype", "teacher_cosine"]
    contiguous_rows: tuple[int, ...]


class SparseRetrievalFunnel:
    """Student-proposed, teacher-reranked retrieval over packed state."""

    def __init__(self, proposal: CosineTeacherStudentProposal) -> None:
        self.proposal = proposal
        self._packed: PackedStateLayout | None = None

    def build_packed_layout(self) -> PackedStateLayout:
        self._packed = PackedStateLayout.from_items(self.proposal.prototypes.state_embeddings.items())
        return self._packed

    @property
    def packed_layout(self) -> PackedStateLayout:
        if self._packed is None:
            raise RuntimeError("packed layout has not been built")
        return self._packed

    def lookup(
        self,
        query: Query,
        *,
        beam_width: int | None = None,
        max_shortlist: int | None = None,
    ) -> SparseFunnelLookup:
        if self._packed is None:
            self.build_packed_layout()
        result = self.proposal.lookup(query, beam_width=beam_width, max_shortlist=max_shortlist)
        rows = self.packed_layout.rows_for_addresses(result.candidate_addresses)
        layout = self.packed_layout.layout_diagnostics(result.candidate_addresses)
        return SparseFunnelLookup(
            proposal=result,
            layout=layout,
            selected_row=(
                self.packed_layout.row(result.selected_address)
                if result.selected_address is not None
                else None
            ),
            addressing_mode="prototype",
            contiguous_rows=rows,
        )
