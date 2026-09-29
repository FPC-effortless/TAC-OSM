"""Integrated selective loop with explicit temporal and retrieval boundaries.

This module is the runtime hardening path. It keeps the v0 router as an
injected component, but changes the call graph to:

    address -> retain -> route -> execute -> observe -> verify -> repair -> learn

The key properties are:
  * writes and reads occur on distinct enforced time boundaries;
  * retrieval is an actual runtime input to routing and execution;
  * query-time addressing can be bounded after index construction;
  * address, readout, representation, and routing are separate interfaces;
  * trajectory and cost accounting are produced by the loop itself;
  * verifier evidence is structured and repair re-executes under a fixed bound;
  * candidate generation supports unique, multiple-valid, and no-valid cases.

The default executor is a synthetic relation executor, explicitly named as a
control. A real CASM adapter can be injected without changing the loop.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

from . import Candidate, Computation, Outcome, Query, RoutingDecision, StateUpdate, Structure
from .addressing import AddressCost, ContentAddressIndex, SUPPORTED_RELATIONS
from .benchmark_v1 import GeneratedTask, RelationName, ValidityMode, generate_task, relation_holds
from .structured_verifier import (
    BoundedExecutableRepair,
    RelationConstraintVerifier,
    StructuredRepair,
    VerificationEvidence,
)
from .temporal import TemporalPersistentState, TemporalWrite
from .trajectory import Trajectory, TrajectoryStep

Router = Callable[[Query, Any, Sequence[Candidate]], RoutingDecision]


@dataclass
class RuntimeCosts:
    """Observed runtime work, kept separate from modelled legacy costs."""

    index_build_candidates: int = 0
    address_query_positions: int = 0
    candidates_available: int = 0
    candidates_routed: int = 0
    executor_invocations: int = 0
    verifier_checks: int = 0
    repair_attempts: int = 0
    wall_clock_seconds: float = 0.0

    def merge(self, other: "RuntimeCosts") -> None:
        for name in (
            "index_build_candidates",
            "address_query_positions",
            "candidates_available",
            "candidates_routed",
            "executor_invocations",
            "verifier_checks",
            "repair_attempts",
        ):
            setattr(self, name, getattr(self, name) + getattr(other, name))
        self.wall_clock_seconds += other.wall_clock_seconds

    def to_dict(self) -> dict[str, Any]:
        return {
            "index_build_candidates": self.index_build_candidates,
            "address_query_positions": self.address_query_positions,
            "candidates_available": self.candidates_available,
            "candidates_routed": self.candidates_routed,
            "executor_invocations": self.executor_invocations,
            "verifier_checks": self.verifier_checks,
            "repair_attempts": self.repair_attempts,
            "wall_clock_seconds": self.wall_clock_seconds,
        }


@dataclass(frozen=True)
class TemporalTaskEvent:
    step: int
    task: GeneratedTask
    world_write: TemporalWrite | None = None


class TemporalBenchmark:
    """Deterministic task stream with separate world writes and decision steps."""

    def __init__(
        self,
        *,
        seed: int = 0,
        dim: int = 8,
        n_candidates: int = 64,
        relation: RelationName = "equality",
        validity: ValidityMode = "unique",
        state: TemporalPersistentState | None = None,
    ) -> None:
        self.seed = seed
        self.dim = dim
        self.n_candidates = n_candidates
        self.relation = relation
        self.validity = validity
        self.state = state if state is not None else TemporalPersistentState()
        self._events: dict[int, TemporalTaskEvent] = {}
        self._writes: dict[int, list[tuple[str, tuple[int, ...], int]]] = {}
        self._last_task_step = -1

    def schedule_lookup_probe(
        self,
        *,
        write_step: int = 0,
        delay: int = 4,
        key: str = "persistent:probe",
        value: Sequence[int] | None = None,
        read_step: int | None = None,
    ) -> GeneratedTask:
        if delay < 1:
            raise ValueError("delay must be >= 1 for an intervening-boundary probe")
        if read_step is None:
            read_step = write_step + delay
        if read_step - write_step != delay:
            raise ValueError("read_step must equal write_step + delay")
        bits = (
            tuple(int(x) for x in value)
            if value is not None
            else tuple((self.seed + j) % 2 for j in range(self.dim))
        )
        if len(bits) != self.dim:
            raise ValueError("value length must equal dim")

        task = generate_task(
            self.seed + read_step,
            dim=self.dim,
            n_candidates=self.n_candidates,
            relation=self.relation,
            validity=self.validity,
            state_address=key,
            reference_bits=bits,
            step=read_step,
        )
        self._writes.setdefault(write_step, []).append((key, bits, read_step))
        self._events[read_step] = TemporalTaskEvent(step=read_step, task=task)
        for step in range(write_step, read_step):
            if step not in self._events:
                filler = generate_task(
                    self.seed + 100_000 + step,
                    dim=self.dim,
                    n_candidates=self.n_candidates,
                    relation=self.relation,
                    validity=self.validity,
                    step=step,
                )
                self._events[step] = TemporalTaskEvent(step=step, task=filler)
        return task

    def schedule_task(
        self,
        step: int,
        *,
        state_address: str = "",
        reference_bits: Sequence[int] | None = None,
    ) -> GeneratedTask:
        task = generate_task(
            self.seed + 200_000 + step,
            dim=self.dim,
            n_candidates=self.n_candidates,
            relation=self.relation,
            validity=self.validity,
            state_address=state_address,
            reference_bits=reference_bits,
            step=step,
        )
        self._events[step] = TemporalTaskEvent(step=step, task=task)
        return task

    def next_task(self, step: int) -> GeneratedTask:
        """Expose exactly one causal decision boundary at a time."""
        expected = self._last_task_step + 1
        if step != expected:
            raise ValueError(
                f"temporal benchmark requires contiguous decision steps: "
                f"expected {expected}, got {step}"
            )
        for write_step, entries in tuple(self._writes.items()):
            if write_step != step:
                continue
            for key, bits, read_step in entries:
                self.state.stage_world_write(
                    StateUpdate(key=key, value=bits, step=write_step),
                    delay=read_step - write_step,
                )
            del self._writes[write_step]
        self.state.advance_to(step)
        self._last_task_step = step
        event = self._events.get(step)
        if event is None:
            return self.schedule_task(step)
        return event.task

    @staticmethod
    def observe(task: GeneratedTask, action: int) -> Outcome:
        success = action in task.acceptable_actions
        return Outcome(
            success=success,
            value=1.0 if success else 0.0,
            feedback="acceptable" if success else "not_acceptable",
            detail=None,
        )


class RelationExecutor:
    """Synthetic structural executor control; replaceable by a CASM adapter."""

    def execute(
        self,
        candidate: Candidate,
        reference: Sequence[int],
        context: Sequence[int],
        relation: RelationName,
    ) -> Computation:
        checks = tuple(
            int(candidate.descriptor[j] == reference[j])
            for j in range(min(len(candidate.descriptor), len(reference), len(context)))
            if context[j]
        )
        output = 1.0 if relation_holds(
            relation,
            reference,
            candidate.descriptor,
            context,
        ) else 0.0
        return Computation(
            structure=Structure(
                key=candidate.key,
                spec={"kind": "synthetic_relation_control", "relation": relation},
                provenance="hardened_v1.synthetic_executor",
            ),
            action=candidate.action,
            trace=(output,) + tuple(float(x) for x in checks),
        )


def _safe_digest(mapping: dict[str, Any]) -> str:
    raw = repr(sorted(mapping.items())).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]


def _population_key(candidates: Sequence[Candidate], context: Sequence[int]) -> str:
    raw = repr(
        (
            tuple((c.key, tuple(c.descriptor)) for c in candidates),
            tuple(context),
        )
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


@dataclass
class HardenedLoop:
    """Single call graph for exhaustive and indexed runtime arms."""

    router: Router
    benchmark: TemporalBenchmark
    index: ContentAddressIndex | None = None
    index_k: int | None = None
    verifier: RelationConstraintVerifier = field(
        default_factory=RelationConstraintVerifier
    )
    repair: BoundedExecutableRepair | None = field(
        default_factory=lambda: BoundedExecutableRepair(max_attempts=3)
    )
    executor: RelationExecutor = field(default_factory=RelationExecutor)
    episode_id: str = "episode-0"

    def __post_init__(self) -> None:
        self.trajectory = Trajectory(self.episode_id)
        self.costs = RuntimeCosts()
        self._index_key: str | None = None
        self._address_cost = AddressCost(
            build_candidates=0, query_positions=0, candidates_returned=0
        )

    def build_index(self, candidates: Sequence[Candidate], context: Sequence[int]) -> None:
        """Build or rebuild only when the indexed population actually changes."""
        start = time.perf_counter()
        self.index = ContentAddressIndex.build(candidates, context=context)
        self.costs.index_build_candidates += len(candidates)
        self.costs.wall_clock_seconds += time.perf_counter() - start
        self._index_key = _population_key(candidates, context)

    def _reference(self, task: GeneratedTask) -> tuple[int, ...]:
        """Use public query bits or addressed state, never hidden gold."""
        read = self.benchmark.state.read(task.public())
        if read.values:
            return tuple(read.values[0])
        bits = task.query.text.partition("	")[0]
        if bits.strip():
            return tuple(int(x) for x in bits.split())
        return ()

    def _state_snapshot(self) -> dict[str, Any]:
        return {
            "step": self.benchmark.state.current_step,
            "available_addresses": self.benchmark.state.addresses(),
            "pending_writes": len(self.benchmark.state.pending()),
        }

    def _state_digest(self, snapshot: dict[str, Any]) -> str:
        return _safe_digest(snapshot)

    def step(self, step: int) -> TrajectoryStep:
        start = time.perf_counter()
        task = self.benchmark.next_task(step)
        public_query = task.public()
        state_before_snapshot = self._state_snapshot()
        state_before = self._state_digest(state_before_snapshot)
        read = self.benchmark.state.read(public_query)
        reference = self._reference(task)
        if not reference:
            raise RuntimeError(
                "no reference available: a persistence query before its write "
                "must remain unreadable rather than being guessed"
            )

        if self.index is not None:
            if task.relation not in SUPPORTED_RELATIONS:
                raise ValueError(
                    f"indexed runtime arm only supports {SUPPORTED_RELATIONS}; "
                    f"got {task.relation!r}"
                )
            key = _population_key(task.candidates, public_query.context)
            if key != self._index_key:
                self.build_index(task.candidates, public_query.context)

        if self.index is None:
            retained = tuple(range(len(task.candidates)))
            address_positions = 0
            bucket_size = len(retained)
            self.costs.candidates_available += len(retained)
        else:
            hit = self.index.lookup(
                public_query,
                reference=reference,
                k=self.index_k,
                relation=task.relation,
            )
            retained = hit.candidate_indices
            address_positions = hit.inspected_positions
            bucket_size = hit.bucket_size
            self.costs.address_query_positions += address_positions
            self.costs.candidates_available += len(retained)

        if not retained:
            raise RuntimeError(
                "retrieval returned an empty set; this is an explicit retrieval "
                "failure, not an invitation to fall back to exhaustive scoring"
            )

        retained_candidates = tuple(task.candidates[i] for i in retained)
        self.costs.candidates_routed += len(retained_candidates)
        decision = self.router(public_query, self.benchmark.state, retained_candidates)
        if not 0 <= decision.selected < len(retained_candidates):
            raise IndexError(
                f"router selected {decision.selected} from "
                f"{len(retained_candidates)} candidates"
            )
        selected_global = retained[decision.selected]

        candidate = task.candidates[selected_global]
        computation = self.executor.execute(
            candidate,
            reference,
            public_query.context,
            task.relation,
        )
        self.costs.executor_invocations += 1
        outcome = self.benchmark.observe(task, selected_global)
        evidence = self.verifier.verify_task(
            computation,
            outcome,
            candidate=candidate,
            reference=reference,
            context=public_query.context,
            relation=task.relation,
        )
        self.costs.verifier_checks += 1
        final_evidence = evidence

        repair_result: StructuredRepair | None = None
        final_selected = selected_global
        final_outcome = outcome
        if not evidence.valid and self.repair is not None:
            action_to_candidate = {
                candidate.action: candidate for candidate in retained_candidates
            }

            def compute_alt(obj: object) -> Computation:
                c = obj if isinstance(obj, Candidate) else candidate
                self.costs.executor_invocations += 1
                return self.executor.execute(
                    c, reference, public_query.context, task.relation
                )

            def verify_alt(comp: Computation) -> VerificationEvidence:
                alt_action = comp.action
                alt_outcome = self.benchmark.observe(task, alt_action)
                alt_candidate = action_to_candidate.get(alt_action)
                if alt_candidate is None:
                    raise RuntimeError(
                        f"repair produced unknown candidate action {alt_action}"
                    )
                self.costs.verifier_checks += 1
                return self.verifier.verify_task(
                    comp,
                    alt_outcome,
                    candidate=alt_candidate,
                    reference=reference,
                    context=public_query.context,
                    relation=task.relation,
                )

            local_repair = self.repair.repair(
                retained_candidates,
                compute=compute_alt,
                verify=verify_alt,
                selected_index=decision.selected,
            )
            repair_result = local_repair
            self.costs.repair_attempts += local_repair.attempts
            if local_repair.passed and local_repair.selected_index is not None:
                final_local = local_repair.selected_index
                final_selected = retained[final_local]
                final_outcome = self.benchmark.observe(task, final_selected)
                final_evidence = local_repair.verification

        if final_evidence.valid and final_outcome.success:
            key = f"experience:{self.episode_id}:{step}"
            self.benchmark.state.stage_world_write(
                StateUpdate(
                    key=key,
                    value=tuple(reference),
                    step=step,
                ),
                delay=1,
            )

        state_after_snapshot = self._state_snapshot()
        state_after = self._state_digest(state_after_snapshot)
        elapsed = time.perf_counter() - start
        self.costs.wall_clock_seconds += elapsed

        query_record = {
            "text": public_query.text,
            "context": tuple(public_query.context),
            "step": public_query.step,
            "provenance": public_query.provenance,
        }
        retrieval_record = {
            "total_candidates": len(task.candidates),
            "retained_candidates": tuple(retained),
            "retained_count": len(retained),
            "address_query_positions": address_positions,
            "bucket_size": bucket_size,
            "index_built": self.index is not None,
        }
        observation_record = {
            "success": bool(final_outcome.success),
            "value": final_outcome.value,
            "feedback": final_outcome.feedback,
        }
        verification_record = {
            "valid": final_evidence.valid,
            "failed_constraint": final_evidence.failed_constraint,
            "counterexample": final_evidence.counterexample,
            "repair_target": final_evidence.repair_target,
            "confidence": final_evidence.confidence,
            "evidence": final_evidence.evidence,
        }
        repair_record = None if repair_result is None else {
            "attempts": repair_result.attempts,
            "passed": repair_result.passed,
            "selected_index": repair_result.selected_index,
            "patch": repair_result.patch,
        }

        row = TrajectoryStep(
            episode_id=self.episode_id,
            step=step,
            state_before={**state_before_snapshot, "digest": state_before},
            query=query_record,
            retrieval=retrieval_record,
            selected=final_selected,
            computation={
                "structure": computation.structure.key,
                "trace_len": len(computation.trace),
            },
            action=final_selected,
            observation=observation_record,
            verification=verification_record,
            repair=repair_record,
            state_after={**state_after_snapshot, "digest": state_after},
            learning={
                "state_write_staged": bool(evidence.valid and final_outcome.success),
                "update_delay": 1 if evidence.valid and final_outcome.success else None,
            },
            provenance={
                "relation": task.relation,
                "validity": task.validity,
                "runtime_cost": {
                    "candidates_routed": len(retained_candidates),
                    "executor_invocations": self.costs.executor_invocations,
                    "verifier_checks": self.costs.verifier_checks,
                },
                "router": getattr(decision, "provenance", "unknown"),
            },
        )
        self.trajectory.append(row)
        return row

    def run(self, n_steps: int) -> Trajectory:
        if n_steps < 1:
            raise ValueError("n_steps must be >= 1")
        for step in range(n_steps):
            self.step(step)
        return self.trajectory
