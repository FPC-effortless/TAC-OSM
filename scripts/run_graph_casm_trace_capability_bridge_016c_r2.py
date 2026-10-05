#!/usr/bin/env python3
"""G-CASM-016C-R2: corrected trace-to-verified-capability bridge.

R2 supersedes R1. R1's scalar arm was structurally empty: candidate evidence was
an int while target evidence was a 1-tuple, so ``int == (int,)`` was always
False and every scalar compatible bucket had size 0. R2 fixes the type match and
enforces it as a runner-level invariant on every generated trial, together with
an explicit non-emptiness condition -- the target is always a member of its own
compatible bucket, so an empty bucket can only arise from a type mismatch.
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import resource
import statistics
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import run_graph_casm_structured_action_probe_015 as g15
from tac_osm.trace_capability import (
    assert_evidence_invariant,
    budget_capped_utility,
    compatible_bucket,
    shortlist,
)

EXPERIMENT_ID = "TACOSM-GRAPH-CASM-TRACE-CAPABILITY-BRIDGE-016C-R2"
SEEDS = (0, 1, 2, 3, 4)
M_LEVELS = (32, 64, 128, 256, 512)
TASKS_PER_M = 32
PRIMARY_M = 512
BUDGET = 8
CHANNELS = ("scalar_row", "activation_trace")
INPUT_ROWS = g15.INPUT_ROWS
LIBRARY_SIZE = g15.LIBRARY_SIZE
GENERATOR_COMMIT = g15.GENERATOR_COMMIT

# Exposed so offline tests can tell whether the CASM generator source is
# importable on this machine. On the Termux control plane it is not -- the
# generator lives in third_party and needs torch -- so tests that require it
# skip locally and run on the CI runner.
GENERATOR_AVAILABLE = bool(getattr(g15, "GENERATOR_AVAILABLE", True))


def verify_shortlist(target, candidates, indices):
    work = 0
    verified_candidates = 0
    for idx in indices:
        ok = True
        candidate = candidates[idx]
        verified_candidates += 1
        for bits in INPUT_ROWS:
            got, cost = g15.g010.execute_exact(candidate, bits)
            work += int(cost.total)
            if got != int(target.truth_table[bits]):
                ok = False
                break
        if ok:
            return 1.0, work, verified_candidates
    return 0.0, work, verified_candidates


def candidate_evidence(cache, indices, channel, row):
    """Candidate evidence values for one selected probe row.

    Both channels yield bare values -- an int for scalar_row, a tuple of ints for
    activation_trace -- so the target evidence must be extracted the same way.
    """
    if channel == "scalar_row":
        return tuple(cache.scalar[(idx, row)][0] for idx in indices)
    return tuple(cache.trace[(idx, row)][0] for idx in indices)


def target_evidence(cache, task, channel, row):
    """Target evidence for one selected probe row, same type as each candidate.

    The R1 bug was here: scalar target evidence was wrapped in a 1-tuple while
    candidate evidence was a bare int, making the compatible bucket empty by
    construction. Extract the stored value the same way in both arms.
    """
    if channel == "scalar_row":
        return cache.scalar[(task.target_index, row)][0]
    return cache.trace[(task.target_index, row)][0]


def assert_evidence_invariant_generated(evidence, target, channel, row):
    """Thin wrapper over the library invariant with the runner's diagnostics.

    Kept as a named symbol so the runner's call sites stay explicit and so an
    offline test can assert the runner enforces the invariant on generated
    evidence. The conditions themselves live in
    ``tac_osm.trace_capability`` so they are testable without torch.
    """
    assert_evidence_invariant(evidence, target, channel, row)


def channel_trial(cache, task, candidates, channel):
    indices = tuple(range(len(candidates)))
    # action_scores builds a signature over all M candidates for each candidate
    # probe row, so the real acquisition scan is O(M * rows). Count it rather
    # than charging only the selected row's per-candidate mean.
    scores = g15.action_scores(cache, indices, channel)
    candidate_scan_operations = len(indices) * len(INPUT_ROWS)
    signature_construction_operations = candidate_scan_operations

    best = min(
        scores,
        key=lambda score: (
            -score.information_per_work,
            -score.information_gain_bits,
            score.expected_remaining_candidates,
            score.action.kind,
            score.action.parameters,
        ),
    )
    row = int(best.action.parameters[0])

    evidence = candidate_evidence(cache, indices, channel, row)
    target = target_evidence(cache, task, channel, row)
    assert_evidence_invariant_generated(evidence, target, channel, row)

    bucket = compatible_bucket(evidence, target)
    selected = shortlist(bucket, BUDGET)
    target = candidates[task.target_index]
    success, verifier_work, verified_candidates = verify_shortlist(target, candidates, selected)

    exhaustive_work = 0
    for candidate in candidates:
        for bits in INPUT_ROWS:
            _, cost = g15.g010.execute_exact(candidate, bits)
            exhaustive_work += int(cost.total)

    # The recorded probe work must be a TOTAL over candidates, not the
    # per-candidate mean that 015 reports, otherwise the accounting scales as
    # O(1) while the real scan is O(M).
    probe_work_total = float(best.expected_cost) * len(indices)
    probe_selection_operations = 1

    return {
        "selected_row": row,
        "information_gain_bits": float(best.information_gain_bits),
        "expected_remaining_candidates": float(best.expected_remaining_candidates),
        "information_per_work": float(best.information_per_work),
        "signature_count": int(best.signature_count),
        "target_bucket_size": len(bucket),
        "shortlist_size": len(selected),
        "target_in_shortlist": float(task.target_index in selected),
        "verified_success": float(success),
        "budget_capped_utility": budget_capped_utility(evidence, BUDGET),
        "probe_environment_work_units_per_candidate": float(best.expected_cost),
        "probe_environment_work_units": probe_work_total,
        "verifier_work_units": int(verifier_work),
        "verified_candidates": int(verified_candidates),
        "exhaustive_reference_work_units": int(exhaustive_work),
        "verified_execution_work_fraction": verifier_work / max(1, exhaustive_work),
        "accounted_total_work_units": probe_work_total + verifier_work,
        "success_per_accounted_work": float(success) / max(1.0, probe_work_total + verifier_work),
        "signature_construction_operations": int(signature_construction_operations),
        "candidate_scan_operations": int(candidate_scan_operations),
        "probe_selection_operations": int(probe_selection_operations),
        "execution_operations": int(verifier_work),
        "verification_operations": int(verifier_work),
        "total_operations": int(
            signature_construction_operations
            + candidate_scan_operations
            + probe_selection_operations
            + verifier_work
        ),
    }


def summarize(raw, seeds, m_levels):
    by_m = {}
    for m in m_levels:
        trials_by_channel = {c: [] for c in CHANNELS}
        seed_success = {c: [] for c in CHANNELS}
        for seed in seeds:
            for c in CHANNELS:
                vals = [task[c] for task in raw[str(seed)][str(m)]]
                trials_by_channel[c].extend(vals)
                seed_success[c].append(statistics.fmean(v["verified_success"] for v in vals))
        by_m[str(m)] = {
            "trials": len(trials_by_channel["scalar_row"]),
            "channels": {},
        }
        for c in CHANNELS:
            vals = trials_by_channel[c]
            by_m[str(m)]["channels"][c] = {
                "verified_success_mean": statistics.fmean(v["verified_success"] for v in vals),
                "budget_capped_utility_mean": statistics.fmean(v["budget_capped_utility"] for v in vals),
                "probe_environment_work_units_per_candidate_mean": statistics.fmean(v["probe_environment_work_units_per_candidate"] for v in vals),
                "probe_environment_work_units_mean": statistics.fmean(v["probe_environment_work_units"] for v in vals),
                "verifier_work_units_mean": statistics.fmean(v["verifier_work_units"] for v in vals),
                "verified_candidates_mean": statistics.fmean(v["verified_candidates"] for v in vals),
                "accounted_total_work_units_mean": statistics.fmean(v["accounted_total_work_units"] for v in vals),
                "success_per_accounted_work_mean": statistics.fmean(v["success_per_accounted_work"] for v in vals),
                "target_bucket_size_mean": statistics.fmean(v["target_bucket_size"] for v in vals),
                "information_gain_bits_mean": statistics.fmean(v["information_gain_bits"] for v in vals),
                "expected_remaining_candidates_mean": statistics.fmean(v["expected_remaining_candidates"] for v in vals),
                "information_per_work_mean": statistics.fmean(v["information_per_work"] for v in vals),
                "verified_execution_work_fraction_mean": statistics.fmean(v["verified_execution_work_fraction"] for v in vals),
                "signature_construction_operations_mean": statistics.fmean(v["signature_construction_operations"] for v in vals),
                "candidate_scan_operations_mean": statistics.fmean(v["candidate_scan_operations"] for v in vals),
                "probe_selection_operations_mean": statistics.fmean(v["probe_selection_operations"] for v in vals),
                "execution_operations_mean": statistics.fmean(v["execution_operations"] for v in vals),
                "verification_operations_mean": statistics.fmean(v["verification_operations"] for v in vals),
                "total_operations_mean": statistics.fmean(v["total_operations"] for v in vals),
                "verified_success_per_total_operations_mean": statistics.fmean(
                    v["verified_success"] / max(1.0, float(v["total_operations"])) for v in vals
                ),
                "seed_verified_success_means": seed_success[c],
            }

    deltas = []
    primary_available = str(PRIMARY_M) in raw[str(seeds[0])]
    if primary_available:
        for seed in seeds:
            trace = statistics.fmean(v["activation_trace"]["verified_success"] for v in raw[str(seed)][str(PRIMARY_M)])
            scalar = statistics.fmean(v["scalar_row"]["verified_success"] for v in raw[str(seed)][str(PRIMARY_M)])
            deltas.append(trace - scalar)

    # Bootstrap the paired five-seed effect.
    import random
    if deltas:
        rng = random.Random(16063)
        draws = sorted(
            statistics.fmean(deltas[rng.randrange(len(deltas))] for _ in deltas)
            for _ in range(4000)
        )
        ci = [
            float(draws[int(.025 * (len(draws)-1))]),
            float(draws[int(.975 * (len(draws)-1))]),
        ]
    else:
        ci = None
    return {
        "by_m": by_m,
        "primary": ({
            "M": PRIMARY_M,
            "B": BUDGET,
            "seed_level_paired_delta": deltas,
            "mean_paired_delta": statistics.fmean(deltas),
            "seed_bootstrap_95ci": ci,
        } if deltas else {
            "M": PRIMARY_M,
            "B": BUDGET,
            "available": False,
            "reason": "smoke omitted primary M; no confirmatory inference",
        }),
    }


def main(smoke=False):
    from tac_osm.contract import load_contract
    contract = load_contract(EXPERIMENT_ID)
    if smoke:
        seeds, m_levels, tasks_per_m = (0,), (32, 128), 4
    else:
        seeds, m_levels, tasks_per_m = SEEDS, M_LEVELS, TASKS_PER_M
        contract.require_levels(M_LEVELS)
        contract.require_seeds(SEEDS)
        contract.require_steps(1)
        contract.require_eval_steps(TASKS_PER_M)
        contract.require_arms(CHANNELS)

    raw = {str(seed): {str(m): [] for m in m_levels} for seed in seeds}
    checks = {}

    for seed in seeds:
        training = g15.g010.generate(seed + 100, g15.TRAIN_PROGRAMS)
        train_s = {g15.g010.structure_key(ep) for ep in training}
        train_t = {g15.g010.truth_signature(ep) for ep in training}
        library = g15.g010.generate(
            seed + 5000, LIBRARY_SIZE,
            exclude_structures=train_s,
            exclude_truths=train_t,
        )
        if len(library) != LIBRARY_SIZE:
            raise RuntimeError("evaluation library construction failed")
        library_s = {g15.g010.structure_key(ep) for ep in library}
        library_t = {g15.g010.truth_signature(ep) for ep in library}
        if train_s & library_s or train_t & library_t:
            raise RuntimeError("train/eval leakage gate failed")

        cache, _ = g15.build_cache(library)
        checks[str(seed)] = {
            "train_eval_structure_disjoint": True,
            "train_eval_truth_disjoint": True,
            "target_identity_available_to_selector": False,
            "target_index_available_to_selector": False,
            "target_evidence_available_before_selection": False,
            "verifier_labels_available_to_selector": False,
            "generator_commit": GENERATOR_COMMIT,
        }

        for m in m_levels:
            candidates = library[:m]
            for task_i in range(tasks_per_m):
                task = g15.build_query_task(seed, m, task_i)
                raw[str(seed)][str(m)].append({
                    c: channel_trial(cache, task, candidates, c)
                    for c in CHANNELS
                })

    summary = summarize(raw, seeds, m_levels)
    result = {
        "experiment_id": EXPERIMENT_ID,
        "status": "measured",
        "provenance": {
            "tacosm_commit": os.environ.get("GITHUB_SHA", "local"),
            "run_id": os.environ.get("GITHUB_RUN_ID", "local"),
            "generator_commit": GENERATOR_COMMIT,
        },
        "protocol": {
            "seeds": list(seeds),
            "M_levels": list(m_levels),
            "tasks_per_seed_M": tasks_per_m,
            "B": BUDGET,
            "channels": list(CHANNELS),
            "primary_M": PRIMARY_M,
        },
        "checks": checks,
        "results": raw,
        "summary": summary,
        "scope": {
            "finite_domain_capability_bridge": True,
            "learned_probe_policy_claim": False,
            "global_optimal_sequential_policy_claim": False,
            "asymptotic_claim": False,
            "language_image_audio_claim": False,
        },
        "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
    }
    out = Path("artifacts")
    out.mkdir(exist_ok=True)
    (out / f"{EXPERIMENT_ID}.json").write_text(json.dumps(result, indent=2, sort_keys=True))
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true")
    main(p.parse_args().smoke)
