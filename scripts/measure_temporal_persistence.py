#!/usr/bin/env python
"""TACOSM-TEMPORAL-001 — temporal persistence across enforced boundaries.

This is a mechanism/integration measurement on the hardened runtime. It does
not train a model and therefore does not invoke the learned-weight integrity
gate. The state-conditioned router is a deterministic control that makes the
temporal variable observable: with carry it reads the world write; with reset
or corrupt the persistent representation is removed or altered at the read
boundary.

The probe is always:

    world write at t -> k intervening decision boundaries -> read at t+k

The hidden task reference belongs only to the environment/executor. Routing
can see only the public query and whatever representation the persistent state
returns at the read boundary.
"""
from __future__ import annotations

import argparse
import os
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from tac_osm import Candidate, RoutingDecision
from tac_osm.contract import load_contract
from tac_osm.hardened import HardenedLoop, TemporalBenchmark
from tac_osm.measurement import results as _results
from tac_osm.measurement.results import Design, Gate, GateCell, MeasurementRecord, Provenance
from tac_osm.measurement.verdicts import seed_spread
from tac_osm.temporal import TemporalPersistentState

EXPERIMENT_ID = "TACOSM-TEMPORAL-001"
ARMS = ("carry", "reset", "corrupt")
DELAYS = (1, 2, 4, 8, 16, 32)
SEEDS = (0, 1, 2, 3, 4)
H = 64
STEPS = 33
EVAL_STEPS = 100


class StateEqualityRouter:
    """Deterministic state-conditioned router control.

    The router never receives hidden task truth. It reads an addressed value
    from persistent state. If no value is available, it falls back to the
    first candidate, which is a chance-level no-memory control because
    candidate order is randomized by the task generator.
    """

    def __call__(
        self,
        query,
        state: TemporalPersistentState,
        candidates: Sequence[Candidate],
    ) -> RoutingDecision:
        read = state.read(query)
        if read.values:
            reference = read.values[0]
            scores = tuple(
                float(
                    sum(
                        int(c.descriptor[j] == reference[j])
                        for j in range(min(len(c.descriptor), len(reference), len(query.context)))
                        if query.context[j]
                    )
                )
                for c in candidates
            )
            selected = max(range(len(candidates)), key=lambda i: (scores[i], -i))
        else:
            scores = tuple(0.0 for _ in candidates)
            selected = 0
        return RoutingDecision(
            selected=selected,
            scores=scores,
            provenance="temporal_state_control",
        )


def _provenance() -> Provenance:
    contract = load_contract(EXPERIMENT_ID)
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


def _boundary_gate(seed: int, delay: int) -> GateCell:
    benchmark = TemporalBenchmark(seed=seed, dim=8, n_candidates=H)
    task = benchmark.schedule_lookup_probe(write_step=0, delay=delay)
    for step in range(delay):
        benchmark.next_task(step)
    # The world write exists, but causal availability has not arrived.
    readable = bool(benchmark.state.read(task.public()).values)
    if readable:
        raise RuntimeError(
            f"temporal boundary leaked at seed={seed}, delay={delay}: "
            "state became readable before t+k"
        )
    return GateCell(
        cell=f"seed={seed},delay={delay}",
        metric="pre_read_available",
        published=0.0,
        observed=float(readable),
        diff=float(readable),
        tol=0.0,
        passed=not readable,
    )


def _apply_arm(arm: str, task, step: int, state: TemporalPersistentState) -> None:
    if step != task.query.step:
        return
    _, _, key = task.query.text.partition("\t")
    if arm == "carry":
        return
    if arm == "reset":
        state.clear()
        return
    if arm == "corrupt":
        state.corrupt(key, position=0)
        return
    raise ValueError(f"unknown arm {arm!r}")


@dataclass(frozen=True)
class Cell:
    decision_success: float
    read_available: float
    verification_passed: float
    router_candidates: float


def _run_cell(seed: int, delay: int, arm: str) -> Cell:
    benchmark = TemporalBenchmark(
        seed=seed,
        dim=8,
        n_candidates=H,
        relation="equality",
        validity="unique",
    )
    task = benchmark.schedule_lookup_probe(
        write_step=0,
        delay=delay,
        key=f"memory:{seed}:{delay}",
    )
    loop = HardenedLoop(
        router=StateEqualityRouter(),
        benchmark=benchmark,
        index=None,
        repair=None,
        learning_enabled=False,
        before_read=lambda t, s, st: _apply_arm(arm, t, s, st),
        episode_id=f"temporal-{arm}-{seed}-{delay}",
    )
    trajectory = loop.run(delay + 1)
    row = trajectory.steps[-1]
    readable = bool(benchmark.state.read(task.public()).values)
    return Cell(
        decision_success=float(row.observation["success"]),
        read_available=float(readable),
        verification_passed=float(row.verification["valid"]),
        router_candidates=float(row.retrieval["total_candidates"]),
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=STEPS)
    ap.add_argument("--eval-steps", type=int, default=EVAL_STEPS)
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--levels", default=",".join(map(str, DELAYS)))
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    seeds = tuple(int(x) for x in args.seeds.split(",") if x.strip())
    delays = tuple(int(x) for x in args.levels.split(",") if x.strip())
    contract = load_contract(EXPERIMENT_ID)
    deviations = ()
    if args.smoke:
        deviations = tuple(
            _results.report_smoke(
                contract,
                EXPERIMENT_ID,
                steps=args.steps,
                eval_steps=args.eval_steps,
                h_levels=(H,),
                seeds=seeds,
                arms=ARMS,
            )
        )
    else:
        contract.require_steps(args.steps)
        contract.require_eval_steps(args.eval_steps)
        contract.require_levels((H,))
        contract.require_k_levels(delays)
        contract.require_seeds(seeds)
        contract.require_arms(ARMS)

    gate_cells = tuple(
        _boundary_gate(seed, delay) for seed in seeds for delay in delays
    )
    gate_ok = all(cell.passed for cell in gate_cells)

    endpoints: dict[str, object] = {}
    per_seed: dict[str, object] = {}
    decision: list[dict] = []
    audit_cells: list[dict] = []

    for delay in delays:
        carry = [_run_cell(seed, delay, "carry") for seed in seeds]
        for arm in ARMS:
            cells = carry if arm == "carry" else [
                _run_cell(seed, delay, arm) for seed in seeds
            ]
            endpoints[f"{arm}@k={delay}"] = {
                "decision_success": statistics.fmean(c.decision_success for c in cells),
                "read_available": statistics.fmean(c.read_available for c in cells),
                "verification_passed": statistics.fmean(c.verification_passed for c in cells),
                "router_candidates": statistics.fmean(c.router_candidates for c in cells),
            }
            per_seed[f"{arm}@k={delay}"] = {
                "decision_success": {
                    "seeds": {
                        str(seed): cell.decision_success
                        for seed, cell in zip(seeds, cells)
                    },
                    "mean": statistics.fmean(c.decision_success for c in cells),
                    "spread": seed_spread([c.decision_success for c in cells]),
                },
                "read_available": {
                    "seeds": {
                        str(seed): cell.read_available
                        for seed, cell in zip(seeds, cells)
                    },
                    "mean": statistics.fmean(c.read_available for c in cells),
                    "spread": seed_spread([c.read_available for c in cells]),
                },
            }
        carry_mean = statistics.fmean(c.decision_success for c in carry)
        for arm in ("reset", "corrupt"):
            cells = [_run_cell(seed, delay, arm) for seed in seeds]
            control_mean = statistics.fmean(c.decision_success for c in cells)
            decision.append(
                {
                    "delay": delay,
                    "arm": arm,
                    "carry_decision_success": carry_mean,
                    "control_decision_success": control_mean,
                    "delta_carry_minus_control": carry_mean - control_mean,
                    "control_spread": seed_spread([c.decision_success for c in cells]),
                }
            )
        audit_cells.append(
            {
                "delay": delay,
                "carry_read_available": statistics.fmean(c.read_available for c in carry),
                "reset_read_available": statistics.fmean(
                    _run_cell(seed, delay, "reset").read_available for seed in seeds
                ),
                "corrupt_read_available": statistics.fmean(
                    _run_cell(seed, delay, "corrupt").read_available for seed in seeds
                ),
            }
        )

    record = MeasurementRecord(
        provenance=_provenance(),
        design=Design(
            steps=args.steps,
            eval_steps=args.eval_steps,
            seeds=seeds,
            h_levels=(H,),
            k_levels=delays,
            arms=ARMS,
            smoke=bool(args.smoke),
            contract_checked=not args.smoke,
            deviations=deviations,
        ),
        gate=Gate(
            name="pre-read causal boundary",
            tolerance="state must be unreadable at t+k-1",
            passed=gate_ok,
            cells=gate_cells,
        ),
        endpoints=endpoints,
        decision_rule=tuple(decision),
        audit={"temporal_boundary": audit_cells},
        per_seed={"cells": per_seed},
    )
    out = _results.results_dir_for(__file__) / "temporal_001.json"
    _results.write_record(record, out)
    print(f"machine-readable summary written to {out}")
    print(f"gate={'PASS' if gate_ok else 'FAIL'}")
    for delay in delays:
        for arm in ARMS:
            row = endpoints[f"{arm}@k={delay}"]
            print(
                f"{arm:>8} k={delay:>2} "
                f"success={row['decision_success']:.4f} "
                f"read={row['read_available']:.4f} "
                f"verify={row['verification_passed']:.4f}"
            )


if __name__ == "__main__":
    main()
