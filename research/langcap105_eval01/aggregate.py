"""Publish every frozen model/seed, descriptive paired uncertainty, no promotion."""
import argparse
import json
import math
from pathlib import Path
import statistics

from common import EVAL_PROTOCOL_SHA256, REGISTRATION_COMMIT, digest, load_plan, require


def bootstrap_difference(left, right, settings):
    import numpy as np
    groups = sorted(left[0]["by_group"])
    require(len(groups) == 17, "Expected all 17 original DEV groups")
    a = np.array([[r["by_group"][g]["nll_sum"] for g in groups] for r in left], dtype=np.float64)
    b = np.array([[r["by_group"][g]["nll_sum"] for g in groups] for r in right], dtype=np.float64)
    n = np.array([[r["by_group"][g]["targets"] for g in groups] for r in left], dtype=np.float64)
    require(all(set(r["by_group"]) == set(groups) for r in left + right), "Mismatched DEV groups")
    require(all(left[s]["by_group"][g]["targets"] == right[s]["by_group"][g]["targets"] for s in range(3) for g in groups),
            "Paired denominators differ")
    rng = np.random.default_rng(settings["rng_seed"])
    values = []
    # Resample paired seed IDs, then paired group IDs; share each group draw
    # across sampled seeds and both models to preserve the crossed design.
    for _ in range(settings["replicates"]):
        ss = rng.integers(0, 3, 3); gg = rng.integers(0, 17, 17)
        values.append(float((a-b)[ss[:, None], gg].sum() / n[ss[:, None], gg].sum()))
    interval = np.percentile(values, settings["interval_percentiles"]).tolist()
    return {"mean_paired_nll_difference":statistics.mean(x["nll"]-y["nll"] for x, y in zip(left, right)),
            "descriptive_95_percentile_interval":interval, "replicates":settings["replicates"],
            "sign":"negative favors left frozen checkpoint on this DEV cohort",
            "scope":"paired seed/group resampling within one public source project; not architecture superiority"}


def summary(values):
    return {"mean":statistics.mean(values), "sample_sd":statistics.stdev(values), "minimum":min(values), "maximum":max(values)}


def run(root, output):
    p = load_plan()
    expected = {(r["scale"], r["arm"], r["seed"]):r for r in p["checkpoint_locks"]}
    results = {}; receipts = []
    for path in sorted(root.rglob("RESULT.json")):
        r = json.loads(path.read_text()); key = (r["scale"], r["arm"], r["seed"])
        require(key in expected and key not in results, "Unexpected or duplicate result")
        require(r["status"] == "FROZEN_EXPLORATORY_DEV_SCORED" and r["protocol_sha256"] == EVAL_PROTOCOL_SHA256,
                "Incomplete or changed evaluation")
        require(r["checkpoint_sha256"] == expected[key]["checkpoint_sha256"] and r["frozen_weights_unchanged"] is True,
                "Frozen checkpoint identity differs")
        require(r["optimizer_updates"] == 0 and r["checkpoint_selection"] is False and r["protected_test_access"] is False,
                "Evaluation changed model or heldout contract")
        require(set(r["metrics"]) == {"INITIAL_DEV", "TRAINED_DEV", "TRAINED_FIT_REFERENCE"}, "Missing scoring mode")
        for mode, scores in r["metrics"].items():
            expected_hash = (p["evaluation"]["dev_window_ids_sha256"] if mode != "TRAINED_FIT_REFERENCE"
                             else p["evaluation"]["fit_references_by_seed"][str(key[2])]["window_ids_sha256"])
            require(scores["targets"] == 594688 and scores["window_ids_sha256"] == expected_hash, "Metric cohort changed")
            require(sum(g["targets"] for g in scores["by_group"].values()) == scores["targets"], "Group denominator differs")
            require(sum(g["targets"] for g in scores["by_file"].values()) == scores["targets"], "File denominator differs")
            require(math.isfinite(scores["nll_sum"]) and scores["nll_sum"] >= 0 and 0 <= scores["correct"] <= scores["targets"], "Invalid aggregate score")
            require(math.isclose(scores["nll"], scores["nll_sum"]/scores["targets"], abs_tol=1e-12, rel_tol=1e-12), "NLL arithmetic differs")
            require(math.isclose(scores["accuracy"], scores["correct"]/scores["targets"], abs_tol=1e-12, rel_tol=1e-12), "Accuracy arithmetic differs")
            require(math.isclose(scores["bits_per_byte"], scores["nll"]/math.log(2), abs_tol=1e-12, rel_tol=1e-12), "Bits/byte arithmetic differs")
            require(math.isclose(math.fsum(g["nll_sum"] for g in scores["by_group"].values()), scores["nll_sum"], abs_tol=1e-8, rel_tol=1e-12), "Group loss sums differ")
            require(sum(g["correct"] for g in scores["by_file"].values()) == scores["correct"], "File accuracy counts differ")
        results[key] = r; receipts.append({"key":list(key), "result_sha256":digest(path)})
    require(set(results) == set(expected), "Missing frozen evaluation models: " + str(sorted(set(expected)-set(results))))
    require(not output.exists(), "Refuse overwrite of previous evaluation summary")
    seeds = [10301, 10302, 10303]; rows = []; pairs = []
    for scale in ("small", "medium"):
        for arm in ("tac_associative_stack", "causal_transformer", "stacked_gru"):
            rs = [results[(scale, arm, seed)] for seed in seeds]
            initial = [r["metrics"]["INITIAL_DEV"] for r in rs]
            dev = [r["metrics"]["TRAINED_DEV"] for r in rs]
            fit = [r["metrics"]["TRAINED_FIT_REFERENCE"] for r in rs]
            rows.append({"scale":scale, "arm":arm, "seeds":seeds,
                "parameters":expected[(scale, arm, seeds[0])]["trainable_and_deployed_parameters"],
                "dev_nll":summary([r["nll"] for r in dev]), "dev_bits_per_byte":summary([r["bits_per_byte"] for r in dev]),
                "dev_accuracy":summary([r["accuracy"] for r in dev]),
                "initial_dev_nll":summary([r["nll"] for r in initial]),
                "fixed_final_fit_reference_nll":summary([r["nll"] for r in fit]),
                "fixed_final_dev_minus_fit_nll":summary([r["fixed_final_dev_minus_fit_reference_nll"] for r in rs]),
                "initial_to_trained_dev_nll_reduction":summary([r["initial_to_trained_dev_nll_reduction"] for r in rs]),
                "learning_transfer_passed_seeds":sum(r["learning_transfers_to_dev_under_registered_criterion"] for r in rs),
                "trained_vs_initial_descriptive_uncertainty":bootstrap_difference(dev, initial, p["evaluation"]["bootstrap"]),
                "per_seed":[{"seed":r["seed"], "dev_nll":r["metrics"]["TRAINED_DEV"]["nll"],
                             "dev_accuracy":r["metrics"]["TRAINED_DEV"]["accuracy"],
                             "fit_reference_nll":r["metrics"]["TRAINED_FIT_REFERENCE"]["nll"]} for r in rs]})
        tac = [results[(scale, "tac_associative_stack", s)]["metrics"]["TRAINED_DEV"] for s in seeds]
        for rival in ("causal_transformer", "stacked_gru"):
            other = [results[(scale, rival, s)]["metrics"]["TRAINED_DEV"] for s in seeds]
            pairs.append({"scale":scale, "left":"tac_associative_stack", "right":rival,
                          **bootstrap_difference(tac, other, p["evaluation"]["bootstrap"])})
    controls = []
    for seed in seeds:
        same = [r["unmatched_diagnostic"] for key, r in results.items() if key[2] == seed]
        require(all(x["exposure_window_ids_sha256"] == same[0]["exposure_window_ids_sha256"] and
            x["fit_target_exposures"] == same[0]["fit_target_exposures"] and
            abs(x["metrics"]["nll"]-same[0]["metrics"]["nll"]) <= 1e-10 and
            x["metrics"]["correct"] == same[0]["metrics"]["correct"] for x in same),
            "Same-seed diagnostic exposure/control differs across models")
        controls.append({"seed":seed, **same[0]})
    report = {"schema":"langcap105.eval01.complete-frozen-dev-summary.v1",
        "status":"ALL_18_FROZEN_MODELS_DEV_SCORED_EXPLORATORY", "model_count":18,
        "protocol_sha256":EVAL_PROTOCOL_SHA256, "registration_commit":REGISTRATION_COMMIT,
        "dev_targets_per_model":594688, "dev_files":18, "dev_groups":17,
        "model_summaries":rows, "descriptive_tac_rival_differences":pairs,
        "unmatched_fit_exposure_diagnostic":controls, "individual_result_receipts":receipts,
        "optimizer_updates":0, "protected_test_access":False, "checkpoint_selection":False,
        "independently_qualified":False, "architecture_superiority_eligible":False, "limits":p["limits"]}
    output.mkdir(parents=True)
    (output / "SUMMARY.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    lines = ["# LANGCAP105-EVAL01 frozen DEV results", "",
        "All 18 original final checkpoints scored; zero optimizer updates. This is public same-project DEV, not independent/sealed TEST.", "",
        "| Scale | Model | DEV NLL | DEV bits/byte | DEV byte accuracy | Final-model seen-FIT NLL | Initial DEV NLL | Learning-transfer seeds |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for r in rows:
        lines.append(f"| {r['scale']} | {r['arm']} | {r['dev_nll']['mean']:.6f} | {r['dev_bits_per_byte']['mean']:.6f} | {100*r['dev_accuracy']['mean']:.3f}% | {r['fixed_final_fit_reference_nll']['mean']:.6f} | {r['initial_dev_nll']['mean']:.6f} | {r['learning_transfer_passed_seeds']}/3 |")
    lines += ["", "Means across all three seeds; NLL denominators and window IDs are identical across models. Per-seed, per-file/group scores and descriptive paired uncertainty are retained in JSON.",
        "", "A lower DEV NLL records better prediction by these frozen checkpoints under this recipe. Training full-FLOPs/tuning parity and independent qualification are incomplete; no architecture is promoted or declared generally superior.",
        "", "The fixed final-model FIT reference differs from the changing-model training-tail losses. The FIT/DEV gap can reflect both memorization and differences between source groups; it does not uniquely identify overfitting.",
        "", "Throughput is teacher-forced CPU batch scoring, not autoregressive decode speed. No long-memory, executable coding, reasoning or agent capability was tested."]
    (output / "RESULTS.md").write_text("\n".join(lines)+"\n")
    print(json.dumps({"status":report["status"], "models":18, "optimizer_updates":0}), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__); p.add_argument("--results", type=Path, required=True); p.add_argument("--output", type=Path, required=True)
    a = p.parse_args(); run(a.results, a.output)
