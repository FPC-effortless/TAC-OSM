"""C5 OR-LSH admission adapter for the unified PLM state layer.

This module reuses the already-implemented C5 dense CDL + OR-LSH machinery.
It is an adapter, not a new representation search. The scientific question is
whether that measured admission mechanism transfers when the addressed object
is persistent PLM state rather than a one-shot candidate pool.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import random
import statistics
from typing import Sequence

from . import Query, StateRead
from .c5_admission_scaling_audit import (
    AuditConfig,
    build_population_extended,
    dense_rank,
    empirical_quantile,
    prefix_lookup,
    train_router,
)
from .c5_noisy_full_phase import ORLSHIndex, estimate_p1_p2, derive_tables
from .plm_unified import (
    MemoryRecord,
    StateKind,
    StateStatus,
    TypedPersistentState,
    PLMConfig,
)


@dataclass(frozen=True)
class C5AdmissionResult:
    """Persistent-state admission result with full address work accounting."""

    ids: tuple[str, ...]
    raw_candidates: int
    admitted_candidates: int
    rerank_count: int
    routing_ops: int
    requested_tables: int
    actual_tables: int
    k90: int
    target_admitted: bool


class C5PersistentStateAdmission:
    """Settled C5 OR-LSH mechanism behind the PLM state-address interface."""

    def __init__(
        self,
        *,
        seed: int,
        calibration_m: int = 1024,
        calibration_trials: int = 64,
        k_alpha: float = 0.10,
        lsh_cap: int = 128,
    ) -> None:
        self.seed = seed
        self.calibration_m = calibration_m
        self.router = train_router(seed, AuditConfig())
        self.k_alpha = float(k_alpha)
        self.lsh_cap = int(lsh_cap)

        calibration = [
            self._trial(seed + 7, step=7000 + i, m=calibration_m)
            for i in range(calibration_trials)
        ]
        ranks = [dense_rank(self.router, trial) for trial in calibration]
        self.k90 = max(
            1,
            min(calibration_m, empirical_quantile(ranks, 1.0 - self.k_alpha)),
        )

        p1s = []
        p2s = []
        probe_index = self._build_index(
            calibration_m,
            tables=1,
            seed_suffix=100,
        )
        for trial in calibration[:32]:
            p1, p2 = estimate_p1_p2(
                self.router,
                probe_index,
                trial,
                samples=4,
            )
            p1s.append(p1)
            p2s.append(p2)
        self.p1 = statistics.fmean(p1s)
        self.p2 = statistics.fmean(p2s)
        self.rho, _ = derive_tables(
            self.p1,
            self.p2,
            calibration_m,
        )

        self._indexes: dict[int, ORLSHIndex] = {}
        self._states: dict[int, TypedPersistentState] = {}
        self._tables: dict[int, int] = {}

    def _trial(self, seed: int, *, step: int, m: int):
        from .c5_noisy_full_phase import make_trial
        return make_trial(
            seed=seed,
            step=step,
            candidates=build_population_extended(seed, m),
        )

    def _build_index(
        self,
        m: int,
        *,
        tables: int,
        seed_suffix: int = 0,
    ) -> ORLSHIndex:
        candidates = build_population_extended(self.seed, m)
        index = ORLSHIndex(
            latent_dim=16,
            bits=max(1, math.ceil(math.log2(m))),
            tables=tables,
            cap=self.lsh_cap,
            seed=self.seed * 9176 + m + seed_suffix,
        )
        index.build(self.router.inner.candidate_embeddings(candidates))
        return index

    def _prepare(self, m: int) -> tuple[TypedPersistentState, ORLSHIndex, int]:
        if m in self._indexes:
            return self._states[m], self._indexes[m], self._tables[m]

        tables = max(1, math.ceil(m ** self.rho))
        index = self._build_index(m, tables=tables)
        state = TypedPersistentState(
            PLMConfig(
                consolidation_threshold=max(m + 1, 256),
                seed=self.seed,
            )
        )
        # The persistent records and indexed candidate keys must come from
        # the same population. Mixing populations invalidates returned IDs
        # and can deterministically suppress target admission.
        candidates = build_population_extended(self.seed, m)
        state.load(
            tuple(
                MemoryRecord(
                    record_id=c.key,
                    namespace="relational",
                    key_bits=tuple(c.descriptor),
                    payload=tuple(c.descriptor[:2]),
                    state_kind=StateKind.WORLD,
                    status=StateStatus.VERIFIED,
                    created_step=0,
                    source="c5_persistent_state",
                    confidence=1.0,
                    operator_family="xor",
                )
                for c in candidates
            )
        )
        self._indexes[m] = index
        self._states[m] = state
        self._tables[m] = tables
        return state, index, tables


    def lookup(
        self,
        *,
        query: Query,
        m: int,
        target_ids: Sequence[str],
    ) -> C5AdmissionResult:
        state, index, tables = self._prepare(m)
        del state
        bits, _ = self._parse(query)
        qvec = self.router.inner.encode_query(query, self._empty_temporal())
        position = {c.key: c.key for c in state.records()}

        raw_addresses, raw_count, hash_ops = prefix_lookup(
            index,
            qvec,
            tables=min(tables, index.actual_tables),
            k=min(self.k90, m),
            score=lambda key, qz=qvec: self.router.inner.score_embeddings(
                qz, index.embeddings[key]
            ),
        )
        ids = tuple(key for key in raw_addresses if key in position)
        target_set = set(target_ids)
        return C5AdmissionResult(
            ids=ids,
            raw_candidates=raw_count,
            admitted_candidates=len(ids),
            rerank_count=raw_count,
            routing_ops=hash_ops + raw_count * index.latent_dim,
            requested_tables=tables,
            actual_tables=index.actual_tables,
            k90=self.k90,
            target_admitted=bool(target_set.intersection(ids)),
        )

    @staticmethod
    def _parse(query: Query) -> tuple[tuple[int, ...], str]:
        bits, _, address = query.text.partition("\t")
        return (
            tuple(int(x) for x in bits.split()) if bits.strip() else (),
            address,
        )

    @staticmethod
    def _empty_temporal():
        from .temporal import TemporalPersistentState
        return TemporalPersistentState()

    def target_records(self, m: int):
        state, _, _ = self._prepare(m)
        return state.records()


def make_persistent_query(
    descriptor: Sequence[int],
    *,
    seed: int,
    step: int,
) -> Query:
    rng = random.Random(seed * 1009 + step)
    noisy = tuple(
        1 - int(bit) if rng.random() < 0.0625 else int(bit)
        for bit in descriptor
    )
    return Query(
        text=" ".join(str(int(x)) for x in noisy),
        context=tuple(1 for _ in descriptor),
        step=step,
        provenance="c5_persistent_state_query",
    )
