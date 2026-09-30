#!/usr/bin/env python3
"""Run TACOSM-C5-SPARSE-FUNNEL-001."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tac_osm import Query
from tac_osm.c5_end_to_end import EndToEndExecutor, build_task, prepare_state
from tac_osm.contract import load_contract
from tac_osm.learned_prototype_state import LearnedPrototypeStateIndex, PrototypeStateIndexConfig
from tac_osm.learned_state_index import LearnedSemanticStateIndex, LearnedStateIndexConfig
from tac_osm.noisy_state_tasks import CODEBOOK
from tac_osm.sparse_funnel import SparseRetrievalFunnel
from tac_osm.teacher_student_proposal import CosineTeacherStudentProposal, DistillationConfig

SEEDS = (0, 1, 2, 3, 4)
H_LEVELS = (64, 128, 256)
K_LEVELS = (4, 8, 16)
BEAM_FOR_K = {4: 1, 8: 2, 16: 4}
TRAIN_CODES = tuple(CODEBOOK[16:64])
M = 64
R = 4
LATENT_DIM = 16
TEACHER_EPOCHS = 512
DISTILL_EPOCHS = 256
EVAL_STEPS = 100
NEGATIVE_COUNT = 8
PROTOTYPE_COUNT = 16
BUCKET_CAPACITY = 4
QUERY_PROJECTION_MACS = 10 * LATENT_DIM
EXHAUSTIVE_STATE_SCORE_MACS = M * LATENT_DIM


def make_teacher(seed: int) -> LearnedSemanticStateIndex:
    model = LearnedSemanticStateIndex(LearnedStateIndexConfig(
        input_dim=10, latent_dim=LATENT_DIM, learning_rate=0.02, margin=0.25,
        epochs=TEACHER_EPOCHS, bucket_bits=8, probe_radius=1, shortlist_k=1, seed=seed,
    ))
    updates = model.train_with_negative_coverage(
        TRAIN_CODES, negative_count=NEGATIVE_COUNT, aggregation="mean", positive_views=1
    )
    expected = TEACHER_EPOCHS * len(TRAIN_CODES)
    if updates != expected:
        raise AssertionError(f"unexpected teacher update count: {updates} != {expected}")
    return model


def training_queries(seed: int) -> tuple[Query, ...]:
    queries: list[Query] = []
    for i, code in enumerate(TRAIN_CODES):
        bits = list(code)
        bits[(seed * 3 + i) % len(bits)] ^= 1
        queries.append(Query(
            text=" ".join(str(int(x)) for x in bits),
            step=0,
            provenance="c5_sparse_funnel_distillation_train",
        ))
    return tuple(queries)


def exhaustive_cosine(teacher, task):
    q_raw = teacher.encode_query(task.query)
    q_norm = math.sqrt(sum(x * x for x in q_raw))
    if q_norm <= 1e-8:
        raise AssertionError("teacher query embedding is zero")
    q = tuple(x / q_norm for x in q_raw)
    scored = []
    for update in task.state_updates:
        z_raw = teacher.encode_state(update.value)
        z_norm = math.sqrt(sum(x * x for x in z_raw))
        if z_norm <= 1e-8:
            raise AssertionError("teacher state embedding is zero")
        z = tuple(x / z_norm for x in z_raw)
        scored.append((sum(a * b for a, b in zip(q, z)), update.key, tuple(update.value)))
    scored.sort(key=lambda item: (-item[0], item[1]))
    score, address, value = scored[0]
    return address, value, score, QUERY_PROJECTION_MACS + M * LATENT_DIM


def run_seed(seed: int, h_levels: tuple[int, ...], steps: int) -> list[dict]:
    first = build_task(seed, 64, 0)
    state = prepare_state(first)
    teacher = make_teacher(seed)
    prototype = LearnedPrototypeStateIndex(
        teacher,
        PrototypeStateIndexConfig(
            prototype_count=PROTOTYPE_COUNT, bucket_capacity=BUCKET_CAPACITY, kmeans_iterations=8
        ),
    )
    build_diag = prototype.build(state, TRAIN_CODES)
    train_code_set = set(TRAIN_CODES)
    teacher_addresses = tuple(
        update.key for update in first.state_updates
        if tuple(update.value) in train_code_set
    )
    if len(teacher_addresses) != len(TRAIN_CODES):
        raise AssertionError("distillation state pool must match training-code population")
    proposal = CosineTeacherStudentProposal(
        teacher, prototype,
        DistillationConfig(
            learning_rate=0.02, epochs=DISTILL_EPOCHS,
            teacher_temperature=0.10, student_temperature=0.10,
            beam_width=2, max_shortlist=8, seed=seed,
        ),
        teacher_state_addresses=teacher_addresses,
    )
    distill = proposal.fit(training_queries(seed))
    funnel = SparseRetrievalFunnel(proposal)
    funnel.build_packed_layout()
    executor = EndToEndExecutor()
    results = []
    for h in h_levels:
        accum = {k: {"ex_target": [], "proposal_retention": [], "sel_target": [],
                      "ex_success": [], "sel_success": [], "shortlist": [],
                      "macs": [], "runs": [], "density": []} for k in K_LEVELS}
        for step in range(steps):
            task = build_task(seed, h, step)
            ex_addr, ex_value, _, _ = exhaustive_cosine(teacher, task)
            ex_target = int(ex_addr == task.target_address)
            ex_relevant = tuple(i for i, c in enumerate(task.candidates) if tuple(c.descriptor) == ex_value)
            ex_out, _, _ = executor.execute_population(tuple(task.candidates[i] for i in ex_relevant), ex_value)
            ex_success = int(ex_out == task.expected_output)
            for k in K_LEVELS:
                hit = funnel.lookup(task.query, beam_width=BEAM_FOR_K[k], max_shortlist=k)
                selected = hit.proposal.selected_address
                selected_value = next((tuple(u.value) for u in task.state_updates if u.key == selected), None)
                retention = int(task.target_address in hit.proposal.candidate_addresses)
                selected_target = int(selected == task.target_address)
                if selected_value is None:
                    success = 0
                else:
                    relevant = tuple(i for i, cand in enumerate(task.candidates) if tuple(cand.descriptor) == selected_value)
                    out, _, _ = executor.execute_population(tuple(task.candidates[i] for i in relevant), selected_value)
                    success = int(out == task.expected_output)
                acc = accum[k]
                acc["ex_target"].append(ex_target)
                acc["proposal_retention"].append(retention)
                acc["sel_target"].append(selected_target)
                acc["ex_success"].append(ex_success)
                acc["sel_success"].append(success)
                acc["shortlist"].append(len(hit.proposal.candidate_addresses))
                acc["macs"].append(hit.proposal.total_macs)
                acc["runs"].append(hit.layout["contiguous_runs"])
                acc["density"].append(hit.layout["packing_density"])
        cells = {}
        for k in K_LEVELS:
            v = accum[k]
            cells[str(k)] = {
                "k": k,
                "beam_width": BEAM_FOR_K[k],
                "exhaustive_target_recall": statistics.fmean(v["ex_target"]),
                "proposal_target_retention": statistics.fmean(v["proposal_retention"]),
                "selective_target_recall": statistics.fmean(v["sel_target"]),
                "exhaustive_end_to_end_success": statistics.fmean(v["ex_success"]),
                "selective_end_to_end_success": statistics.fmean(v["sel_success"]),
                "shortlist_size_mean": statistics.fmean(v["shortlist"]),
                "total_macs_mean": statistics.fmean(v["macs"]),
                "arithmetic_reduction": 1.0 - statistics.fmean(v["macs"]) / (QUERY_PROJECTION_MACS + EXHAUSTIVE_STATE_SCORE_MACS),
                "contiguous_runs_mean": statistics.fmean(v["runs"]),
                "packing_density_mean": statistics.fmean(v["density"]),
            }
        results.append({
            "seed": seed, "H": h, "eval_steps": steps,
            "teacher": {"epochs": TEACHER_EPOCHS, "negative_count": NEGATIVE_COUNT},
            "prototype_build": {
                "total_build_macs": build_diag.total_build_macs,
                "max_bucket_size": build_diag.max_bucket_size,
                "min_bucket_size": build_diag.min_bucket_size,
            },
            "distillation": {
                "epochs": distill.epochs, "training_queries": distill.training_queries,
                "optimizer_updates": distill.optimizer_updates,
                "initial_kl": distill.initial_kl, "final_kl": distill.final_kl,
                "teacher_candidate_score_macs": distill.teacher_candidate_score_macs,
                "student_prototype_score_macs": distill.student_prototype_score_macs,
            },
            "cells": cells,
            "exhaustive_query_macs": QUERY_PROJECTION_MACS + EXHAUSTIVE_STATE_SCORE_MACS,
        })
    return results

def run(smoke: bool) -> dict:
    seeds = (0,) if smoke else SEEDS
    levels = (64,) if smoke else H_LEVELS
    steps = 5 if smoke else EVAL_STEPS
    results = []
    for seed in seeds:
        results.extend(run_seed(seed, levels, steps))
    pooled: dict[str, dict] = {}
    for k in K_LEVELS:
        values = [r["cells"][str(k)] for r in results]
        pooled[str(k)] = {
            "exhaustive_target_recall_mean": statistics.fmean(v["exhaustive_target_recall"] for v in values),
            "proposal_target_retention_mean": statistics.fmean(v["proposal_target_retention"] for v in values),
            "selective_target_recall_mean": statistics.fmean(v["selective_target_recall"] for v in values),
            "exhaustive_end_to_end_success_mean": statistics.fmean(v["exhaustive_end_to_end_success"] for v in values),
            "selective_end_to_end_success_mean": statistics.fmean(v["selective_end_to_end_success"] for v in values),
            "shortlist_size_mean": statistics.fmean(v["shortlist_size_mean"] for v in values),
            "total_macs_mean": statistics.fmean(v["total_macs_mean"] for v in values),
            "arithmetic_reduction_mean": statistics.fmean(v["arithmetic_reduction"] for v in values),
            "contiguous_runs_mean": statistics.fmean(v["contiguous_runs_mean"] for v in values),
            "packing_density_mean": statistics.fmean(v["packing_density_mean"] for v in values),
        }
    return {"protocol": {"name": "TACOSM-C5-SPARSE-FUNNEL-001", "smoke": smoke,
             "seeds": list(seeds), "H_levels": list(levels), "K_levels": list(K_LEVELS),
             "beam_for_K": {str(k): BEAM_FOR_K[k] for k in K_LEVELS}, "M": M, "R": R,
             "latent_dim": LATENT_DIM, "teacher_epochs": TEACHER_EPOCHS,
             "distill_epochs": DISTILL_EPOCHS, "negative_count": NEGATIVE_COUNT,
             "prototype_count": PROTOTYPE_COUNT, "bucket_capacity": BUCKET_CAPACITY,
             "eval_steps": steps}, "pooled": pooled, "cells": results}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--no-contract", action="store_true")
    args = parser.parse_args()
    if not args.no_contract:
        contract = load_contract("TACOSM-C5-SPARSE-FUNNEL-001")
        contract.require_levels(H_LEVELS)
        contract.require_seeds(SEEDS)
        contract.require_k_levels(K_LEVELS)
    result = run(args.smoke)
    out = Path("artifacts/TACOSM-C5-SPARSE-FUNNEL-001.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
