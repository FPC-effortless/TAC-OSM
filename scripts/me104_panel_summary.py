"""No-fit descriptive summary and source-label audit for the ME104 panel."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from scripts import me100_sensor_data as data
from scripts.me102_panel_summary import _check_metrics, _metric_row, _numeric_replay
from scripts.me104_split_panel import OUT, RECIPES, SEEDS

PARENT = ROOT / "research/replay-bundles/ME103"
DEST = OUT / "PANEL_SUMMARY.json"
COHORTS = ("gap", "general")
BOOTSTRAP_REPS = 2000
BOOTSTRAP_SEED = 104104
RTOL, ATOL = 1e-5, 1e-6


def _load(path):
    if not path.is_file():
        raise FileNotFoundError(f"saved panel artifact is not available: {path}")
    return json.loads(path.read_text())


def _add_width_quantiles(metric, quantiles, observed):
    width = quantiles[..., 2] - quantiles[..., 0]
    for channel in range(6):
        selected = width[observed[:, channel], channel]
        row = metric["by_channel"][str(channel)]
        row["median_width"] = float(np.quantile(selected, 0.50))
        row["p90_width"] = float(np.quantile(selected, 0.90))
        row["p99_width"] = float(np.quantile(selected, 0.99))
    return metric


def _mean_metrics(rows):
    keys = ("mae", "coverage", "interval_score", "mean_width", "p95_width",
            "median_width", "p90_width", "p99_width")
    return {
        "nll": float(np.mean([row["nll"] for row in rows])),
        "by_channel": {
            str(channel): {key: float(np.mean([row["by_channel"][str(channel)][key] for row in rows]))
                          for key in keys}
            for channel in range(6)
        },
    }


def _cluster_bootstrap(candidate, parent, truth, observed, origins):
    """Paired five-seed PM2.5 MAE difference, resampling global calendar weeks."""
    weeks = np.asarray(origins, dtype=np.int64)[:, 1] // 168
    clusters = np.unique(weeks)
    expected = 21 if len(origins) == 143 else 53
    if len(clusters) != expected:
        raise RuntimeError(f"calendar-week cluster count changed: expected {expected}, got {len(clusters)}")
    cand_error = np.abs(candidate[:, :, 0, 1] - truth[None, :, 0])
    parent_error = np.abs(parent[:, :, 0, 1] - truth[None, :, 0])
    keep = observed[None, :, 0]

    def delta(sampled_weeks):
        row_ix = np.concatenate([np.flatnonzero(weeks == week) for week in sampled_weeks])
        selected = np.broadcast_to(keep[:, row_ix], (candidate.shape[0], len(row_ix)))
        counts = selected.sum(axis=1)
        if np.any(counts == 0):
            raise RuntimeError("bootstrap sample has no observed PM2.5 labels")
        c = np.where(selected, cand_error[:, row_ix], 0).sum(axis=1) / counts
        p = np.where(selected, parent_error[:, row_ix], 0).sum(axis=1) / counts
        return float((c - p).mean())

    point = delta(clusters)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    samples = np.asarray([delta(rng.choice(clusters, size=len(clusters), replace=True))
                          for _ in range(BOOTSTRAP_REPS)])
    ci = np.quantile(samples, (0.025, 0.975))
    return dict(candidate_minus_parent_pm25_mae=point, ci95=ci.tolist(),
                bootstrap_replicates=BOOTSTRAP_REPS, week_count=len(clusters),
                cluster_unit="global calendar week (origin_hour // 168)",
                matched_seed_count=candidate.shape[0],
                interpretation="conditional posthoc descriptive interval; not independent evidence or an acceptance gate")


def summary():
    child = _load(OUT / "RESULTS.json")
    parent_result = _load(PARENT / "RESULTS.json")
    child_index = _load(OUT / "FIT_INDEX.json")
    parent_index = _load(PARENT / "FIT_INDEX.json")
    new = np.load(OUT / "FORECASTS.npz", allow_pickle=False)
    old = np.load(PARENT / "FORECASTS.npz", allow_pickle=False)
    report = data.load("report")

    expected_pairs = {(recipe, seed) for recipe in RECIPES for seed in SEEDS}
    if {(row["architecture"], row["seed"]) for row in child_index["models"]} != expected_pairs:
        raise RuntimeError("ME104 index is not the complete fixed 15-fit panel")
    if len(child_index["models"]) != 15:
        raise RuntimeError("ME104 index contains duplicate or missing fits")
    parent_rows = {(row["recipe"], row["seed"]): row
                   for row in parent_result["cohorts"]["gap"]["models"]
                   if row["recipe"] == "strong_reconstruction"}
    for seed in SEEDS:
        if ("strong_reconstruction", seed - 2000) not in parent_rows:
            raise RuntimeError("missing matched ME103 strong_reconstruction parent")

    cohort_summary = {}
    bootstraps = {}
    source = data.load("report")
    for cohort, expected_n, expected_weeks in (("gap", 143, 21), ("general", 4288, 53)):
        for suffix in ("origins", "truth", "observed", "locf", "ridge"):
            key = f"{cohort}_{suffix}"
            np.testing.assert_array_equal(new[key], old[key], err_msg=f"ME104/ME103 {key} mismatch")
        origins = new[f"{cohort}_origins"]
        truth = new[f"{cohort}_truth"]
        observed = new[f"{cohort}_observed"].astype(bool, copy=False)
        np.testing.assert_array_equal(origins, np.asarray(source[cohort]))
        raw_truth = source["raw"][origins[:, 0], origins[:, 1] + 6, :6]
        np.testing.assert_array_equal(observed, np.isfinite(raw_truth))
        np.testing.assert_array_equal(truth, np.nan_to_num(raw_truth, nan=0))
        weeks = np.unique(origins[:, 1] // 168)
        if len(origins) != expected_n or len(weeks) != expected_weeks:
            raise RuntimeError(f"{cohort} query or calendar-week count changed")

        recipes, candidate_by_recipe, parent_by_recipe = {}, {}, {}
        for recipe in RECIPES:
            candidate_rows, parent_rows_for_recipe = [], []
            candidate_q, parent_q = [], []
            for seed in SEEDS:
                key = f"{cohort}_{recipe}_{seed}"
                parent_key = f"{cohort}_strong_reconstruction_{seed - 2000}"
                q, pq = new[key], old[parent_key]
                for label, values in (("ME104", q), ("ME103 parent", pq)):
                    if values.shape != (expected_n, 6, 3) or not np.isfinite(values).all() or (values < 0).any() or (np.diff(values, axis=-1) < 0).any():
                        raise RuntimeError(f"invalid {label} quantiles for {cohort}/{recipe}/{seed}")
                c_saved = next(r for r in child["cohorts"][cohort]["models"]
                               if r["recipe"] == recipe and r["seed"] == seed)
                p_saved = next(r for r in parent_result["cohorts"][cohort]["models"]
                               if r["recipe"] == "strong_reconstruction" and r["seed"] == seed - 2000)
                cm = _add_width_quantiles(_metric_row(q, truth, observed, c_saved["raw_density_nll"]), q, observed)
                pm = _add_width_quantiles(_metric_row(pq, truth, observed, p_saved["raw_density_nll"]), pq, observed)
                _check_metrics(cm, c_saved, f"ME104 {cohort} {recipe} {seed}")
                _check_metrics(pm, p_saved, f"ME103 strong_reconstruction {cohort} {seed - 2000}")
                candidate_rows.append(dict(seed=seed, **cm))
                parent_rows_for_recipe.append(dict(seed=seed, parent_seed=seed - 2000, **pm))
                candidate_q.append(q)
                parent_q.append(pq)
            candidate_by_recipe[recipe] = np.stack(candidate_q)
            parent_by_recipe[recipe] = np.stack(parent_q)
            gate_rows = [r for r in child["model_gates"] if r["recipe"] == recipe]
            if len(gate_rows) != 5:
                raise RuntimeError(f"expected five saved gate rows for {recipe}")
            recipes[recipe] = dict(
                candidate_seed_count=len(candidate_rows), parent_seed_count=len(parent_rows_for_recipe),
                full_gate_seed_count=sum(bool(r["full_gate"]) for r in gate_rows),
                full_gate_seeds=[r["seed"] for r in gate_rows if r["full_gate"]],
                all_five_gate=child["recipe_all_seed_gates"][recipe],
                candidate_mean_across_five_seeds=_mean_metrics(candidate_rows),
                parent_mean_across_five_seeds=_mean_metrics(parent_rows_for_recipe),
                candidate_per_seed=candidate_rows, parent_per_seed=parent_rows_for_recipe,
                old_parent_full_gate_seed_count=sum(bool(r["full_gate"]) for r in parent_result["models"]
                                                    if r["recipe"] == "strong_reconstruction" and r["seed"] >= 102001 and r["seed"] <= 102005),
            )
            bootstraps[cohort + "/" + recipe] = _cluster_bootstrap(
                candidate_by_recipe[recipe], parent_by_recipe[recipe], truth, observed, origins
            )
        cohort_summary[cohort] = dict(
            query_count=len(origins), calendar_week_cluster_count=len(weeks),
            observed_label_count_by_channel=observed.sum(axis=0).astype(int).tolist(),
            truth_sha256=hashlib.sha256(truth.tobytes()).hexdigest(), recipes=recipes,
        )

    strong_parent_gates = [r for r in parent_result["models"]
                           if r["recipe"] == "strong_reconstruction" and 102001 <= r["seed"] <= 102005]
    me103_all_gates = sum(bool(r["full_gate"]) for r in parent_result["models"])
    result = dict(
        status="EXPOSED2015_POSTHOC_NO_INDEPENDENT_CLAIM",
        purpose="no-fit descriptive comparison of all ME104 recipes with same-seed ME103 strong_reconstruction parents",
        no_fit=True, source_sha256=data.SHA, raw_source_labels_recomputed=True,
        protected_2016_2017_scored=False, independent_qualification=False,
        original_forecast_arrays_modified=False,
        metric_replay_tolerance={"rtol": RTOL, "atol": ATOL},
        gates=dict(me104_full_gate_seed_count=sum(bool(r["full_gate"]) for r in child["model_gates"]),
                   me104_total_seed_count=15,
                   me104_per_recipe={recipe: sum(bool(r["full_gate"]) for r in child["model_gates"] if r["recipe"] == recipe)
                                     for recipe in RECIPES},
                   me103_strong_reconstruction_full_gate_seed_count=sum(bool(r["full_gate"]) for r in strong_parent_gates),
                   me103_strong_reconstruction_seed_count=5,
                   me103_all_recipes_full_gate_seed_count=me103_all_gates,
                   me103_all_recipes_seed_count=len(parent_result["models"])),
        cohorts=cohort_summary,
        paired_pm25_mae_week_cluster_bootstrap=bootstraps,
        bootstrap_note="Conditional posthoc descriptive intervals; gap has 21 global calendar-week clusters and general has 53 overlapping global calendar-week clusters, reported separately and not summed.",
    )
    if DEST.exists():
        _numeric_replay(result, _load(DEST))
        print("ME104_PANEL_SUMMARY_REPLAY=PASS")
    else:
        with DEST.open("x") as stream:
            json.dump(result, stream, indent=2, allow_nan=False)
            stream.write("\n")
        print("ME104_PANEL_SUMMARY_SAVED=PASS")
    for recipe in RECIPES:
        summary_row = cohort_summary["gap"]["recipes"][recipe]
        mae = summary_row["candidate_mean_across_five_seeds"]["by_channel"]["0"]["mae"]
        parent_mae = summary_row["parent_mean_across_five_seeds"]["by_channel"]["0"]["mae"]
        print(json.dumps(dict(recipe=recipe, candidate_pm25_mae=mae, parent_pm25_mae=parent_mae,
                              full_gate_seed_count=summary_row["full_gate_seed_count"],
                              all_five=summary_row["all_five_gate"])))


if __name__ == "__main__":
    summary()
