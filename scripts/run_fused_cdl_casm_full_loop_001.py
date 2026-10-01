"""TACOSM-FUSED-CDL-CASM-001.

Runs the first end-to-end fused information/computation loop:

    persistent state
      -> CDL-style Q/K relevance student
      -> top-1 action with K-admission diagnostics
      -> explicit CASM computation selector
      -> exact structural execution
      -> post-execution verifier
      -> verifier-derived CDL update
      -> verified experience write

The experiment is intentionally modest and auditable. It does not claim C5:
the CDL student still scores all candidates. The purpose is to establish that
the two previously separate validated ideas can participate in one causal loop
without corrupting the state, verification, or leakage boundaries.
"""

from __future__ import annotations

import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

from tac_osm.casm_adapter import casm_structure_from_program
from tac_osm.environment import WorldConfig, WorldEnvironment
from tac_osm.executor import ExecutorConfig, StructuralExecutor
from tac_osm.execution_feedback import ExecutionGroundTruthVerifier
from tac_osm.fused_cdl_casm import CDLConfig, CDLStudentRouter, CasmComputationSelector
from tac_osm.model import ModelConfig, TacOsmModel
from tac_osm.router import OracleRouter, RouterConfig, StaticTotalAgreementRouter
from tac_osm.state import PersistentStore, StateConfig


OUT = Path("results")
OUT.mkdir(parents=True, exist_ok=True)


def _finite_mean(rows: list[float]) -> float:
    values = [x for x in rows if math.isfinite(x)]
    return statistics.mean(values) if values else 0.0


def build_model(
    seed: int,
    *,
    router_kind: str,
    learn: bool,
    intervention: str = "persistent",
) -> tuple[TacOsmModel, CDLStudentRouter | None, CasmComputationSelector]:
    state = PersistentStore(
        StateConfig(
            enabled=True,
            write=True,
            intervention=intervention,
            n_slots=256,
            seed=seed,
        )
    )
    environment = WorldEnvironment(
        WorldConfig(
            dim=8,
            n_candidates=8,
            noise=0.10,
            families=("relational", "state_lookup", "replay"),
            seed=seed,
        )
    )

    cdl = None
    if router_kind == "cdl":
        cdl = CDLStudentRouter(
            CDLConfig(
                input_dim=8,
                latent_dim=16,
                learning_rate=0.01,
                margin=0.25,
                admission_k=4,
                seed=seed,
            )
        )
        router = cdl
    elif router_kind == "static":
        router = StaticTotalAgreementRouter(
            RouterConfig(type="static", dim=8, seed=seed)
        )
    elif router_kind == "oracle":
        router = OracleRouter(RouterConfig(type="oracle", dim=8, seed=seed))
    else:
        raise ValueError(router_kind)

    executor = StructuralExecutor(
        ExecutorConfig(type="learned", dim=8, max_nodes=10, seed=seed)
    )
    selector = CasmComputationSelector(max_nodes=10)
    model = TacOsmModel(
        state=state,
        router=router,
        executor=executor,
        verifier=ExecutionGroundTruthVerifier(),
        repair=None,
        environment=environment,
        computation_selector=selector,
        config=ModelConfig(
            n_steps=1,
            learn=learn,
            seed=seed,
            write_on_success=True,
        ),
    )
    return model, cdl, selector


def run_steps(
    seed: int,
    *,
    router_kind: str,
    n_steps: int,
    learn: bool,
    intervention: str = "persistent",
) -> tuple[list[dict], CDLStudentRouter | None, PersistentStore]:
    model, cdl, selector = build_model(
        seed,
        router_kind=router_kind,
        learn=learn,
        intervention=intervention,
    )
    rows = []
    for i in range(n_steps):
        model.config.n_steps = 1
        step = model.step(i)
        row = {
            "step": i,
            "family": step.provenance["family"],
            "success": bool(step.outcome.success),
            "verified": bool(step.verification.passed),
            "wrote": bool(step.write and step.write.committed),
            "router": step.decision.provenance,
            "selected": step.decision.selected,
            "candidate_count": len(step.decision.scores),
            "scores": list(step.decision.scores),
            "casm_active_nodes": (
                selector.last_selection.active_nodes
                if selector.last_selection is not None
                else 0
            ),
            "casm_true_edges": (
                selector.last_selection.true_edges
                if selector.last_selection is not None
                else 0
            ),
            "casm_candidate_edges": (
                selector.last_selection.candidate_edges
                if selector.last_selection is not None
                else 0
            ),
        }
        if cdl is not None:
            try:
                target = int(getattr(step.outcome.detail, "gold_index"))
                diag = cdl.diagnostics_with_target(
                    step.query,
                    model.state,
                    # The state is queried from the same persistent store used
                    # by the model. This call is post-hoc measurement only.
                    model.environment._current.candidates
                    if model.environment._current is not None
                    else (),
                    target,
                )
            except (AttributeError, TypeError, IndexError):
                # The environment clears _current after transition. Use the
                # selected decision scores to compute only the directly
                # observable candidate count here; target-rank is reconstructed
                # in the dedicated replay pass below.
                diag = None
            row["cdl_updates"] = cdl.updates
            row["cdl_admission_k"] = cdl.config.admission_k
            row["cdl_scored"] = len(step.decision.scores)
        rows.append(row)
    return rows, cdl, model.state


def run_seed(seed: int, *, train_steps: int = 180, eval_steps: int = 90) -> dict:
    train_rows, cdl, train_state = run_steps(
        seed,
        router_kind="cdl",
        n_steps=train_steps,
        learn=True,
        intervention="persistent",
    )

    # Freeze parameters by constructing a new router, then copying the learned
    # matrices. Evaluation uses a clean environment/store so no evaluation
    # outcome can change the trained parameters.
    eval_model, eval_cdl, selector = build_model(
        seed + 10_000,
        router_kind="cdl",
        learn=False,
        intervention="persistent",
    )
    assert cdl is not None and eval_cdl is not None
    eval_cdl.wq = [list(row) for row in cdl.wq]
    eval_cdl.wc = [list(row) for row in cdl.wc]
    eval_cdl.bq = list(cdl.bq)
    eval_cdl.bc = list(cdl.bc)

    eval_rows = []
    for i in range(eval_steps):
        step = eval_model.step(i)
        scores = list(step.decision.scores)
        order = sorted(range(len(scores)), key=lambda j: (-scores[j], j))
        target = int(getattr(step.outcome.detail, "gold_index"))
        eval_rows.append(
            {
                "step": i,
                "family": step.provenance["family"],
                "success": bool(step.outcome.success),
                "verified": bool(step.verification.passed),
                "target_rank": order.index(target) + 1,
                "admission_k": min(eval_cdl.config.admission_k, len(order)),
                "target_in_admission": target in order[: eval_cdl.config.admission_k],
                "candidate_count": len(scores),
                "candidates_scored": len(scores),
                "selected": step.decision.selected,
                "casm_active_nodes": selector.last_selection.active_nodes if selector.last_selection else 0,
                "casm_true_edges": selector.last_selection.true_edges if selector.last_selection else 0,
                "casm_candidate_edges": selector.last_selection.candidate_edges if selector.last_selection else 0,
                "state_found": bool(eval_cdl.last_memory and eval_cdl.last_memory.found),
                "state_inspected_slots": (
                    eval_cdl.last_memory.inspected_slots if eval_cdl.last_memory else 0
                ),
                "state_pool_size": eval_cdl.last_memory.pool_size if eval_cdl.last_memory else 0,
            }
        )

    grouped = defaultdict(list)
    for row in eval_rows:
        grouped[row["family"]].append(row)
    family_summary = {}
    for family, rows in grouped.items():
        family_summary[family] = {
            "steps": len(rows),
            "top1": _finite_mean([float(r["success"]) for r in rows]),
            "admission_recall_k": _finite_mean([float(r["target_in_admission"]) for r in rows]),
            "mean_target_rank": _finite_mean([float(r["target_rank"]) for r in rows]),
            "mean_state_inspected_slots": _finite_mean(
                [float(r["state_inspected_slots"]) for r in rows]
            ),
        }

    static_model, _, static_selector = build_model(
        seed + 20_000,
        router_kind="static",
        learn=False,
        intervention="persistent",
    )
    static_rows = [static_model.step(i) for i in range(eval_steps)]
    static_summary = {
        "top1": _finite_mean([float(s.outcome.success) for s in static_rows]),
        "verified": _finite_mean([float(s.verification.passed) for s in static_rows]),
        "casm_active_nodes": _finite_mean(
            [
                float(static_selector.last_selection.active_nodes)
                for _ in static_rows
                if static_selector.last_selection is not None
            ]
        ),
    }

    oracle_model, _, oracle_selector = build_model(
        seed + 30_000,
        router_kind="oracle",
        learn=False,
        intervention="persistent",
    )
    oracle_rows = [oracle_model.step(i) for i in range(eval_steps)]
    oracle_summary = {
        "top1": _finite_mean([float(s.outcome.success) for s in oracle_rows]),
        "verified": _finite_mean([float(s.verification.passed) for s in oracle_rows]),
        "casm_active_nodes": _finite_mean(
            [
                float(oracle_selector.last_selection.active_nodes)
                for _ in oracle_rows
                if oracle_selector.last_selection is not None
            ]
        ),
    }

    reset_model, reset_cdl, reset_selector = build_model(
        seed + 40_000,
        router_kind="cdl",
        learn=False,
        intervention="reset",
    )
    reset_cdl.wq = [list(row) for row in cdl.wq]
    reset_cdl.wc = [list(row) for row in cdl.wc]
    reset_cdl.bq = list(cdl.bq)
    reset_cdl.bc = list(cdl.bc)
    reset_rows = [reset_model.step(i) for i in range(eval_steps)]
    reset_summary = {
        "top1": _finite_mean([float(s.outcome.success) for s in reset_rows]),
        "verified": _finite_mean([float(s.verification.passed) for s in reset_rows]),
        "casm_active_nodes": _finite_mean(
            [
                float(reset_selector.last_selection.active_nodes)
                for _ in reset_rows
                if reset_selector.last_selection is not None
            ]
        ),
    }

    return {
        "seed": seed,
        "train_steps": train_steps,
        "eval_steps": eval_steps,
        "train_success": _finite_mean([float(r["success"]) for r in train_rows]),
        "trained_updates": cdl.updates if cdl else 0,
        "experience_writes": train_state.writes,
        "persistent_occupancy": train_state.occupancy,
        "eval": {
            "overall_top1": _finite_mean([float(r["success"]) for r in eval_rows]),
            "overall_admission_recall_k": _finite_mean(
                [float(r["target_in_admission"]) for r in eval_rows]
            ),
            "mean_target_rank": _finite_mean([float(r["target_rank"]) for r in eval_rows]),
            "mean_casm_active_nodes": _finite_mean(
                [float(r["casm_active_nodes"]) for r in eval_rows]
            ),
            "mean_casm_true_edges": _finite_mean(
                [float(r["casm_true_edges"]) for r in eval_rows]
            ),
            "mean_casm_candidate_edges": _finite_mean(
                [float(r["casm_candidate_edges"]) for r in eval_rows]
            ),
            "mean_state_inspected_slots": _finite_mean(
                [float(r["state_inspected_slots"]) for r in eval_rows]
            ),
            "mean_state_pool_size": _finite_mean(
                [float(r["state_pool_size"]) for r in eval_rows]
            ),
            "families": family_summary,
            "rows": eval_rows,
        },
        "controls": {
            "static": static_summary,
            "oracle": oracle_summary,
            "reset": reset_summary,
        },
    }


def main() -> None:
    seeds = tuple(range(10))
    runs = [run_seed(seed) for seed in seeds]

    pooled = []
    for run in runs:
        pooled.extend(run["eval"]["rows"])
    summary = {
        "seeds": list(seeds),
        "n_runs": len(runs),
        "pooled_eval_steps": len(pooled),
        "fused_cdl_casm_top1": _finite_mean([float(r["success"]) for r in pooled]),
        "fused_cdl_admission_recall_k": _finite_mean(
            [float(r["target_in_admission"]) for r in pooled]
        ),
        "fused_cdl_mean_target_rank": _finite_mean(
            [float(r["target_rank"]) for r in pooled]
        ),
        "fused_cdl_mean_casm_active_nodes": _finite_mean(
            [float(r["casm_active_nodes"]) for r in pooled]
        ),
        "fused_cdl_mean_casm_candidate_edges": _finite_mean(
            [float(r["casm_candidate_edges"]) for r in pooled]
        ),
        "fused_cdl_mean_state_inspected_slots": _finite_mean(
            [float(r["state_inspected_slots"]) for r in pooled]
        ),
        "fused_cdl_mean_state_pool_size": _finite_mean(
            [float(r["state_pool_size"]) for r in pooled]
        ),
        "mean_trained_updates": _finite_mean([float(r["trained_updates"]) for r in runs]),
        "mean_experience_writes": _finite_mean([float(r["experience_writes"]) for r in runs]),
        "control_static_top1": _finite_mean(
            [float(r["controls"]["static"]["top1"]) for r in runs]
        ),
        "control_oracle_top1": _finite_mean(
            [float(r["controls"]["oracle"]["top1"]) for r in runs]
        ),
        "control_reset_top1": _finite_mean(
            [float(r["controls"]["reset"]["top1"]) for r in runs]
        ),
        "non_claims": [
            "CDL runtime scoring remains O(N) over candidates.",
            "Admission recall is not hardware compute reduction.",
            "The experiment does not establish C5 sublinear end-to-end scaling.",
            "The CDL teacher is not executed at runtime; the runtime component is the cheap Q/K student.",
            "CASM is exact and independently verified for the selected computation; learned CASM gating is not claimed here.",
        ],
        "runs": runs,
    }
    (OUT / "fused_cdl_casm_001.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True)
    )

    print("=== TACOSM-FUSED-CDL-CASM-001 ===")
    for key, value in summary.items():
        if key not in {"runs", "non_claims"}:
            print(f"{key}: {value}")
    print(f"result_file: {OUT / 'fused_cdl_casm_001.json'}")


if __name__ == "__main__":
    main()
