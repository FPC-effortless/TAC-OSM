"""State-conditioned wrapper for the REP-003 graph-program router.

The wrapper changes only the source of the five-bit semantic query:
REP-003 encoded it in Query.text; REP-005 reads it from the addressed
persistent state item at query time.
"""

from __future__ import annotations

from typing import Sequence

from . import Candidate, PersistentState, Query, RoutingDecision
from .graph_program_router import GraphProgramRouter, GraphProgramRouterConfig
from .state_addressing import StateAddressor


class PersistentSemanticGraphRouter(GraphProgramRouter):
    """Graph-program router whose query representation comes from state."""

    def __init__(
        self,
        config: GraphProgramRouterConfig | None = None,
        *,
        addressor: StateAddressor | None = None,
    ) -> None:
        super().__init__(config)
        self.addressor = addressor if addressor is not None else StateAddressor()

    def _state_query(self, query: Query, state: PersistentState) -> Query:
        memory = self.addressor.address(query, state)
        if not memory.found:
            raise ValueError(
                f"required semantic state item is unavailable at {memory.address!r}"
            )
        if len(memory.value) != self.config.query_dim:
            raise ValueError(
                f"persistent semantic width {len(memory.value)} != "
                f"query width {self.config.query_dim}"
            )
        return Query(
            text=" ".join(str(int(x)) for x in memory.value) + "	",
            context=(),
            step=query.step,
            provenance="persistent_state",
        )

    def route(
        self,
        query: Query,
        state: PersistentState,
        candidates: Sequence[Candidate],
    ) -> RoutingDecision:
        return super().route(self._state_query(query, state), state, candidates)

    def state_score(
        self,
        query: Query,
        state: PersistentState,
        candidates: Sequence[Candidate],
    ) -> list[float]:
        return super().score(self._state_query(query, state), candidates)

    def diagnostics_with_target(
        self,
        query: Query,
        state: PersistentState,
        candidates: Sequence[Candidate],
        target_index: int,
    ):
        return super().diagnostics_with_target(
            self._state_query(query, state),
            candidates,
            target_index,
        )

    def learn_from_outcome(
        self,
        query: Query,
        state: PersistentState,
        candidates: Sequence[Candidate],
        selected: int,
        *,
        success: bool,
        scores: Sequence[float] | None = None,
    ) -> float:
        return super().learn_from_outcome(
            self._state_query(query, state),
            state,
            candidates,
            selected,
            success=success,
            scores=scores,
        )
