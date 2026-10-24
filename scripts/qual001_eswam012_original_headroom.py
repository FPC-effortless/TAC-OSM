"""Read-only, retrospective, source-aligned ESWAM012 headroom audit.

This does NOT train a model, use independent TEST, certify matched FLOPs, or
establish architectural superiority. Uses frozen original per-case records.
"""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path

ARMS = ("native", "native_short", "gru", "attention", "native_unscoped")
SEEDS = (86001, 86002)
HORIZONS = (10, 30)
ROOT = Path("research/replay-bundles/ESWAM012")


def identity(r):
    return (r["source_original_recording"], r["source_original_frame"],
            r["source_original_actor_id"], r["horizon_frames"])


def xorshift32(seed):
    state = seed & 0xFFFFFFFF
    while True:
        state ^= (state << 13) & 0xFFFFFFFF
        state ^= state >> 17
        state ^= (state << 5) & 0xFFFFFFFF
        state &= 0xFFFFFFFF
        yield state / 4294967296.0


def paired_actor_ci(rows, deltas, seed, trials=5000):
    if len(rows) != len(deltas) or not rows:
        raise ValueError("Unmatched paired cases")
    actors = sorted({r["source_actor_cluster"] for r in rows})
    groups = {a: [i for i, r in enumerate(rows)
                  if r["source_actor_cluster"] == a] for a in actors}
    random_values = xorshift32(seed)
    draws = []
    for _ in range(trials):
        total = count = 0
        for _ in actors:
            selected = actors[int(next(random_values) * len(actors))]
            for i in groups[selected]:
                total += deltas[i]
                count += 1
        draws.append(total / count)
    draws.sort()
    return [draws[int(.025 * trials) - 1], draws[int(.975 * trials) - 1]]


def aligned_cases(files):
    indexes = {}
    for arm, rows in files.items():
        mapping = {identity(r): r for r in rows}
        if len(mapping) != len(rows):
            raise ValueError(f"Duplicate original query in {arm}")
        indexes[arm] = mapping
    original = indexes["native"]
    for arm, index in indexes.items():
        if set(index) != set(original):
            raise ValueError(f"Original cohort mismatch: {arm}")
        for key, row in index.items():
            source = original[key]
            if row["source_original_recording"] != 3:
                raise ValueError("Unexpected recording (not the exposed DEV)")
            for field in ("true_original_future_xy_m", "observed_current_xy_m",
                          "original_fixed_physical_predictions",
                          "source_actor_cluster"):
                if row[field] != source[field]:
                    raise ValueError(f"Original labels/priors differ: {arm}, {key}")
            error = math.dist(row["true_original_future_xy_m"],
                              row["model_predicted_xy_m"])
            if not math.isclose(error, row["l2_meters"]["model"], abs_tol=2e-6):
                raise ValueError(f"Scored original prediction mismatch: {arm}, {key}")
    return {k: {arm: indexes[arm][k] for arm in ARMS} for k in original}


def audit(root=ROOT, trials=5000):
    report = {
        "study": "QUAL001_RETROSPECTIVE_ESWAM012_HEADROOM",
        "source": "publisher-original traffic, previously exposed recording003, ONE intersection",
        "tier": "REAL_EXPOSED_DEV_NO_FIT_NOT_ARCHITECTURE_QUALIFIED",
        "independent_test_used": False,
        "original_optimizer_guard_certified": False,
        "matched_full_training_flops_certified": False,
        "seeds": [],
    }
    for seed in SEEDS:
        files = {arm: json.loads((root / f"{arm}-{seed}-ORIGINAL_DEV_CASES.json").read_text())
                 for arm in ARMS}
        aligned = aligned_cases(files)
        per_seed = {"seed": seed, "horizons": []}
        for horizon in HORIZONS:
            keys = [k for k in aligned if k[3] == horizon]
            rows = [aligned[k]["native"] for k in keys]
            actors = {r["source_actor_cluster"] for r in rows}
            if len(rows) != 266 or len(actors) != 37:
                raise ValueError("Original source case/actor counts changed")
            scores = {arm: [aligned[k][arm]["l2_meters"]["model"] for k in keys]
                      for arm in ARMS}
            mean = {arm: sum(v) / len(v) for arm, v in scores.items()}
            differences = {}
            for arm in ("native_short", "gru", "attention", "native_unscoped"):
                delta = [a - b for a, b in zip(scores["native"], scores[arm])]
                differences[f"native_minus_{arm}"] = {
                    "mean_m": sum(delta) / len(delta),
                    "actor_cluster_bootstrap95_m":
                        paired_actor_ci(rows, delta, 20261024 + seed + horizon, trials),
                }
            regime = {}
            for label, predicate in (
                ("low_speed", lambda speed: speed < .2),
                ("moving", lambda speed: speed >= .2),
            ):
                ids = [i for i, row in enumerate(rows) if
                       predicate(row["current_measured_source_velocity_norm_m_per_s"])]
                regime[label] = {
                    "cases": len(ids),
                    "mean_m": {arm: sum(scores[arm][i] for i in ids) / len(ids)
                               for arm in ARMS},
                }
            per_seed["horizons"].append({
                "horizon": horizon, "cases": len(rows), "actor_tracks": len(actors),
                "mean_model_L2_m": mean, "paired_comparisons": differences,
                "observed_speed_regimes": regime,
            })
        report["seeds"].append(per_seed)
    return report


def self_test():
    from itertools import islice
    a = list(islice(xorshift32(20261024), 10))
    assert len(a) == 10 and all(0 <= x < 1 for x in a)
    row = {"source_original_recording": 3, "source_original_frame": 0,
           "source_original_actor_id": "1", "horizon_frames": 10,
           "source_actor_cluster": "3:1", "true_original_future_xy_m": [1, 1],
           "observed_current_xy_m": [0, 0],
           "model_predicted_xy_m": [0, 0],
           "original_fixed_physical_predictions": {"prior": [0, 0]},
           "l2_meters": {"model": math.sqrt(2)}}
    original = {arm: [row.copy()] for arm in ARMS}
    aligned_cases(original)
    damaged = {arm: [row.copy()] for arm in ARMS}
    damaged["gru"][0]["true_original_future_xy_m"] = [2, 2]
    try:
        aligned_cases(damaged)
        raise AssertionError("Invalid source labels passed")
    except ValueError:
        pass
    assert paired_actor_ci([row], [.2], 42, 100) == [.2, .2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--verify-frozen-means", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        print("QUAL001 software-only tests: PASS")
        return
    output = audit(args.root)
    if args.verify_frozen_means:
        expected = {
            (86001, 10): (.24853726266774284, .2394788134800988),
            (86001, 30): (1.858627341213784, 1.808548253008858),
            (86002, 10): (.24617653703277115, .2342772650176573),
            (86002, 30): (1.866663528513969, 1.7864674224139192),
        }
        for seed_block in output["seeds"]:
            for h in seed_block["horizons"]:
                old, short = expected[(seed_block["seed"], h["horizon"])]
                values = h["mean_model_L2_m"]
                assert abs(values["native"] - old) < 1e-8
                assert abs(values["native_short"] - short) < 1e-8
        print("QUAL001 original observed-case mean and actor alignment verification: PASS")
    blob = json.dumps(output, sort_keys=True, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(blob)
    else:
        print(blob, end="")


if __name__ == "__main__":
    main()
