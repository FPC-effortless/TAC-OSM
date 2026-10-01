#!/usr/bin/env python3
"""Measure TACOSM-C5-ADMISSION-SCALING-AUDIT-001."""
from __future__ import annotations

import json
import statistics
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tac_osm.c5_admission_scaling_audit import (
    AuditConfig,
    aggregate_dense,
    dense_scaling_for_seed,
    local_slopes,
    lsh_sweep_for_seed,
    train_router,
)


CONFIG = AuditConfig()


def main() -> None:
    dense_per_seed = []
    lsh_per_seed = []
    for seed in CONFIG.seeds:
        router = train_router(seed, CONFIG)
        dense_per_seed.append(dense_scaling_for_seed(router, seed, CONFIG))
        lsh_per_seed.append(lsh_sweep_for_seed(router, seed, CONFIG))

    dense = aggregate_dense(dense_per_seed)
    slopes = local_slopes(dense)

    pooled_lsh = []
    for i, tables in enumerate(CONFIG.lsh_tables):
        rows = [seed["results"][i] for seed in lsh_per_seed]
        pooled_lsh.append(
            {
                "tables": tables,
                "M": CONFIG.sweep_m,
                "bits": lsh_per_seed[0]["bits"],
                "K90_mean": statistics.fmean(seed["K90"] for seed in lsh_per_seed),
                "p1_mean": statistics.fmean(seed["p1"] for seed in lsh_per_seed),
                "p2_mean": statistics.fmean(seed["p2"] for seed in lsh_per_seed),
                "theoretical_L90_mean": statistics.fmean(
                    seed["theoretical_L90"] for seed in lsh_per_seed
                ),
                "admission_recall": statistics.fmean(r["admission_recall"] for r in rows),
                "rerank_fraction": statistics.fmean(r["rerank_fraction"] for r in rows),
                "mean_rerank_count": statistics.fmean(r["mean_rerank_count"] for r in rows),
                "mean_routing_ops": statistics.fmean(r["mean_routing_ops"] for r in rows),
                "routing_fraction": statistics.fmean(r["routing_fraction"] for r in rows),
                "cap_bound": any(r["cap_bound"] for r in rows),
                "theoretical_L90_row": any(r["is_theoretical_L90"] for r in rows),
            }
        )

    all_caps_unbound = all(
        seed["index_actual_tables"] < CONFIG.lsh_cap
        and seed["index_actual_tables"] >= max(CONFIG.lsh_tables)
        for seed in lsh_per_seed
    )
    result = {
        "protocol": {
            "name": "TACOSM-C5-ADMISSION-SCALING-AUDIT-001",
            "status": "measured",
            "purpose": [
                "Extend dense-CDL rank scaling to M=2048,4096,8192.",
                "Sweep OR-LSH table count before any representation intervention.",
            ],
            "task_scope": "Synthetic 64-class 16-bit codebook with one-bit public-query corruption.",
            "main_arm_persistence_input": False,
            "hamming_semantic_shortcut_present": True,
            "Hamming_reference_interpretation": (
                "routing solvability reference only; this workload is not a semantic "
                "retrieval benchmark because class relevance is determined by the binary code."
            ),
            "train_steps": CONFIG.train_steps,
            "calibration_trials": CONFIG.calibration_trials,
            "heldout_trials": CONFIG.heldout_trials,
            "seeds": list(CONFIG.seeds),
            "dense_M": list(CONFIG.dense_m_levels),
            "lsh_sweep_M": CONFIG.sweep_m,
            "lsh_tables": list(CONFIG.lsh_tables),
            "lsh_table_cap": CONFIG.lsh_cap,
        },
        "dense_scaling": {
            "pooled": dense,
            "local_P90_exponents": slopes,
            "fit_gamma_all_M": __import__(
                "tac_osm.c5_admission_scaling_audit", fromlist=["fit_power"]
            ).fit_power(
                [r["M"] for r in dense],
                [r["P90_best_valid_rank"] for r in dense],
            ),
        },
        "lsh_sweep": {
            "pooled": pooled_lsh,
            "cap_unbound": all_caps_unbound,
            "per_seed": lsh_per_seed,
        },
        "controls": {
            "theoretical_approximation": (
                "L90 = ceil(log(0.10) / log(1 - p1^b)) under iid table independence; "
                "reported as a diagnostic, not as a guarantee."
            ),
            "selection": (
                "No sweep row is selected by held-out success. Every registered table "
                "count and the calibration-derived theoretical L90 are reported."
            ),
        },
        "interpretation": {
            "rank_vs_recall": (
                "Dense CDL rank growth and LSH admission are separate quantities. "
                "A large K90 indicates increased ranking work; admission recall measures "
                "additional loss introduced by hashing."
            ),
            "persistence": (
                "The main arm has no persistent state input. The audit therefore does not "
                "establish persistence as part of the funnel."
            ),
            "semantic_scope": (
                "The one-bit code benchmark is algorithmically solvable by exact Hamming "
                "lookup, so improvements here cannot be read as learned semantic retrieval."
            ),
        },
    }
    out = Path("artifacts/TACOSM-C5-ADMISSION-SCALING-AUDIT-001.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
