#!/usr/bin/env python
"""TACOSM-SELECTIVE-001 — actual selective-computation boundary.

A single persistent candidate universe is built once per (seed,H) and reused
for 100 queries. The exhaustive arm routes over H candidates every query. The
indexed arm performs one content-address index build, then queries a bounded
signature bucket and routes only the retained candidates.

This experiment measures the runtime boundary and its accounting. The router
and executor are deterministic controls; this is not by itself evidence for
semantic retrieval, a learned addresser, or the full L4 C5 claim.
"""
from __future__ import annotations

import argparse
import os
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from tac_osm import Candidate, RoutingDecision
from tac_osm.addressing import ContentAddressIndex
from tac_osm.benchmark_v1 import (
    repeated_equality_population,
    task_from_population,
)
from tac_osm.contract import load_contract
from tac_osm.hardened import HardenedLoop
from tac_osm.measurement import results as _results
from tac_osm.measurement.results import Design, Gate, GateCell, MeasurementRecord, Provenance
from tac_osm.measurement.verdicts import seed_spread
from tac_osm.temporal import TemporalPersistentState

EXPERIMENT_ID = "TACOSM-SELECTIVE-001"
ARMS = ("exhaustive", "indexed")
H_LEVELS = (8, 64, 256)
K_LEVELS = (2, 4)
SEEDS = (0, 1, 2, 3, 4)
STEPS = 100
EVAL_STEPS = 100
DIM = 8


class StaticPopulationBenchmark:
    """Static-world benchmark: the candidate universe persists across queries."""

    def __init__(self, *, seed: int, candidates: Sequence[Candidate], steps: int) -> None:
        self.seed = seed
        self.candidates = tuple(candidates)
        self.steps = steps
        self.state = TemporalPersistentState()
        self.current_step = -1
        self._references = (
            (0, 0, 0, 0, 0, 0, 0, 0),
            (0, 1, 1, 0, 0, 1, 0, 1),
            (1, 0, 1, 1, 1, 0, 1, 0),
            (1, 1, 0, 1, 0, 1, 1, 0),
        )

    def next_task(self, step: int):
        expected = self.current_step + 1
        if step != expected:
            raise ValueError(f"non-contiguous step: expected {expected}, got {step}")
        self.current_step = step
        self.state.advance_to(step)
        reference = self._references[(step + self.seed) % len(self._references)]
        return task_from_population(
            candidates=self.candidates,
            reference_bits=reference,
            step=step,
            relation="equality",
            validity="multiple",
        )

    @staticmethod
    def observe(task, action: int):
        success = action in task.acceptable_actions
        from tac_osm import Outcome
        return Outcome(
            success=success,
            value=1.0 if success else 0.0,
            feedback="acceptable" if success else "not_acceptable",
            detail=None,
        )


class PublicEqualityRouter:
    """Exact public relation router; no state or hidden truth is required."""

    def __call__(self, query, state, candidates: Sequence[Candidate]) -> RoutingDecision:
        bits = tuple(
            int(x) for x in query.text.partition("\t")[0].split()
        )
        scores = tuple(
            float(
                sum(
                    int(c.descriptor[j] == bits[j])
                    for j in range(min(len(c.descriptor), len(bits), len(query.context)))
                    if query.context[j]
                )
            )
            for c in candidates
        )
        selected = max(range(len(candidates)), key=lambda i: (scores[i], -i))
        return RoutingDecision(
            selected=selected,
            scores=scores,
            provenance="public_equality_control",
        )


@dataclass(frozen=True)
class Cell:
    success_rate: float
    router_candidates_per_step: float
    index_build_candidates: float
    address_query_positions: float
    amortized_candidate_work: float
    executor_invocations: float
    wall_clock_seconds: float


def _provenance() -> Provenance:
    return Provenance(
        experiment_id=EXPERIMENT_ID,
        contract_source=f"contracts/{EXPERIMENT_ID}.json",
        contract_sha256=_results.contract_fingerprint(
            _results.contract_path_for(__file__, EXPERIMENT_ID)
        ),
        git_commit=_results.git_commit(),
        script=Path(__file__).name,
        python=_results._python_version(),
        recorded_at=_results.now(),
    )


def _run(seed: int, h: int, arm: str, k: int) -> Cell:
    population = repeated_equality_population(
        seed,
        dim=DIM,
        n_candidates=h,
        marked_positions=(0, 1),
    )
    benchmark = StaticPopulationBenchmark(seed=seed, candidates=population, steps=STEPS)
    index = None if arm == "exhaustive" else ContentAddressIndex()
    loop = HardenedLoop(
        router=PublicEqualityRouter(),
        benchmark=benchmark,
        index=index,
        index_k=k if arm == "indexed" else None,
        repair=None,
        learning_enabled=False,
        episode_id=f"selective-{arm}-{seed}-H{h}-K{k}",
    )
    start = time.perf_counter()
    trajectory = loop.run(STEPS)
    elapsed = time.perf_counter() - start
    successes = sum(
        1 for row in trajectory.steps if row.observation["success"]
    )
    costs = loop.costs
    amortized = (
        costs.index_build_candidates
        + costs.address_query_positions
        + costs.candidates_routed
    ) / STEPS
    return Cell(
        success_rate=successes / STEPS,
        router_candidates_per_step=costs.candidates_routed / STEPS,
        index_build_candidates=costs.index_build_candidates,
        address_query_positions=costs.address_query_positions,
        amortized_candidate_work=amortized,
        executor_invocations=costs.executor_invocations,
        wall_clock_seconds=elapsed,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=STEPS)
    ap.add_argument("--eval-steps", type=int, default=EVAL_STEPS)
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--levels", default=",".join(map(str, H_LEVELS)))
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    seeds = tuple(int(x) for x in args.seeds.split(",") if x.strip())
    levels = tuple(int(x) for x in args.levels.split(",") if x.strip())
    contract = load_contract(EXPERIMENT_ID)
    deviations = ()
    if args.smoke:
        deviations = tuple(
            _results.report_smoke(
                contract,
                EXPERIMENT_ID,
                steps=args.steps,
                eval_steps=args.eval_steps,
                h_levels=levels,
                seeds=seeds,
                arms=ARMS,
            )
        )
    else:
        contract.require_steps(args.steps)
        contract.require_eval_steps(args.eval_steps)
        contract.require_levels(levels)
        contract.require_k_levels(K_LEVELS)
        contract.require_seeds(seeds)
        contract.require_arms(ARMS)

    endpoints: dict[str, object] = {}
    per_seed: dict[str, object] = {}
    decision: list[dict] = []
    gate_cells: list[GateCell] = []

    for h in levels:
        for k in K_LEVELS:
            ex = [_run(seed, h, "exhaustive", k) for seed in seeds]
            ix = [_run(seed, h, "indexed", k) for seed in seeds]
            ex_mean = statistics.fmean(c.success_rate for c in ex)
            ix_mean = statistics.fmean(c.success_rate for c in ix)
            endpoints[f"exhaustive@H={h},K={k}"] = {
                "success_rate": ex_mean,
                "router_candidates_per_step": statistics.fmean(
                    c.router_candidates_per_step for c in ex
                ),
                "index_build_candidates": statistics.fmean(
                    c.index_build_candidates for c in ex
                ),
                "address_query_positions": statistics.fmean(
                    c.address_query_positions for c in ex
                ),
                "amortized_candidate_work": statistics.fmean(
                    c.amortized_candidate_work for c in ex
                ),
                "executor_invocations": statistics.fmean(
                    c.executor_invocations for c in ex
                ),
                "wall_clock_seconds": statistics.fmean(c.wall_clock_seconds for c in ex),
            }
            endpoints[f"indexed@H={h},K={k}"] = {
                "success_rate": ix_mean,
                "router_candidates_per_step": statistics.fmean(
                    c.router_candidates_per_step for c in ix
                ),
                "index_build_candidates": statistics.fmean(
                    c.index_build_candidates for c in ix
                ),
                "address_query_positions": statistics.fmean(
                    c.address_query_positions for c in ix
                ),
                "amortized_candidate_work": statistics.fmean(
                    c.amortized_candidate_work for c in ix
                ),
                "executor_invocations": statistics.fmean(
                    c.executor_invocations for c in ix
                ),
                "wall_clock_seconds": statistics.fmean(c.wall_clock_seconds for c in ix),
            }
            per_seed[f"H={h},K={k}"] = {
                "exhaustive_success": {
                    "seeds": {str(s): c.success_rate for s, c in zip(seeds, ex)},
                    "mean": ex_mean,
                    "spread": seed_spread([c.success_rate for c in ex]),
                },
                "indexed_success": {
                    "seeds": {str(s): c.success_rate for s, c in zip(seeds, ix)},
                    "mean": ix_mean,
                    "spread": seed_spread([c.success_rate for c in ix]),
                },
            }
            gate_cells.append(
                GateCell(
                    cell=f"H={h},K={k}",
                    metric="indexed_success_rate",
                    published=1.0,
                    observed=ix_mean,
                    diff=ix_mean - 1.0,
                    tol=0.0,
                    passed=ix_mean >= 1.0 - 1e-12,
                )
            )
            decision.append(
                {
                    "h": h,
                    "k": k,
                    "exhaustive_success_rate": ex_mean,
                    "indexed_success_rate": ix_mean,
                    "delta_indexed_minus_exhaustive": ix_mean - ex_mean,
                    "exhaustive_spread": seed_spread([c.success_rate for c in ex]),
                    "indexed_router_candidates_per_step": statistics.fmean(
                        c.router_candidates_per_step for c in ix
                    ),
                    "exhaustive_router_candidates_per_step": statistics.fmean(
                        c.router_candidates_per_step for c in ex
                    ),
                    "index_build_candidates": statistics.fmean(
                        c.index_build_candidates for c in ix
                    ),
                    "amortized_candidate_work": statistics.fmean(
                        c.amortized_candidate_work for c in ix
                    ),
                }
            )

    gate_ok = all(cell.passed for cell in gate_cells)
    record = MeasurementRecord(
        provenance=_provenance(),
        design=Design(
            steps=args.steps,
            eval_steps=args.eval_steps,
            seeds=seeds,
            h_levels=levels,
            k_levels=K_LEVELS,
            arms=ARMS,
            smoke=bool(args.smoke),
            contract_checked=not args.smoke,
            deviations=deviations,
        ),
        gate=Gate(
            name="indexed control preserves exact task success",
            tolerance="success must be 1.0 in the exact equality control",
            passed=gate_ok,
            cells=tuple(gate_cells),
        ),
        endpoints=endpoints,
        decision_rule=tuple(decision),
        audit={
            "cost_boundary": {
                "one_time_index_build": True,
                "queries_per_population": STEPS,
                "router_cost_is_actual_candidate_input_count": True,
                "executor_cost_is_actual_invocation_count": True,
            }
        },
        per_seed={"cells": per_seed},
    )
    out = _results.results_dir_for(__file__) / "selective_001.json"
    _results.write_record(record, out)
    print(f"machine-readable summary written to {out}")
    print(f"gate={'PASS' if gate_ok else 'FAIL'}")
    for row in decision:
        print(
            f"H={row['h']:>3} K={row['k']} "
            f"success={row['indexed_success_rate']:.4f} "
            f"router={row['indexed_router_candidates_per_step']:.2f} "
            f"amortized={row['amortized_candidate_work']:.2f}"
        )


if __name__ == "__main__":
    main()
