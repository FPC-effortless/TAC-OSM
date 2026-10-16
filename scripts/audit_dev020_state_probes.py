"""DEV-020 independent stdlib integrity/aggregation of post-hoc diagnostic JSON.

Rechecks counts, balanced pair identities implied by metrics, expected grids and
predeclared decision rules. Does not reproduce gradients or model features.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

ARMS = ("memoryless", "gru768", "schema41", "tac_joint", "tac_encoder")
SEEDS = (43, 59, 71)
REGIMES = ("short", "long")
PREFIXES = ("32", "128", "512")
LEVELS = ("state_probe", "read_probe")


def validate(data):
    assert data["study"] == "DEV-020"
    assert data["status"] == "POSTHOC_FROZEN_DIAGNOSTIC_NOT_CONFIRMATORY"
    assert not data["smoke"] and data["probe_only"]
    assert data["train_episodes"] == 288 and data["heldout_episodes"] == 96
    assert data["training_domain"] == "dev020-probe-train"
    assert data["test_domain"] == "dev018-eval"
    assert data["seed"] in SEEDS and data["regime"] in REGIMES
    assert tuple(data["arms"]) == ARMS
    for arm, arm_data in data["arms"].items():
        digest = arm_data["checkpoint_sha256"]
        assert isinstance(digest, str) and len(digest) == 64
        assert tuple(arm_data["prefixes"]) == PREFIXES
        for h, metrics in arm_data["prefixes"].items():
            check_metrics(metrics)
            assert math.isfinite(metrics["latent_pair_l2"]) and metrics["latent_pair_l2"] >= 0
            assert math.isfinite(metrics["target_read_pair_l2"]) and metrics["target_read_pair_l2"] >= 0
        for level in LEVELS:
            probe = arm_data[level]
            assert probe["linear_features"] > 0
            check_metrics(probe["heldout"])
            check_metrics(probe["shuffled_label_control"])
    return data


def check_metrics(m):
    for key in ("accuracy", "paired_class_disagreement", "paired_mean_probability_gap"):
        assert math.isfinite(m[key]) and 0 <= m[key] <= 1
    # Balanced paired outcomes: a same-label predicted pair scores exactly 1/2.
    assert abs(m["accuracy"] - .5) <= .5 * m["paired_class_disagreement"] + 1e-6


def summarize(records):
    assert len(records) == len(SEEDS) * len(REGIMES)
    lookup = {(r["seed"], r["regime"]): validate(r) for r in records}
    assert len(lookup) == len(records)
    assert set(lookup) == {(s, r) for s in SEEDS for r in REGIMES}
    result = {"study": "DEV-020", "status": "POSTHOC_NOT_CONFIRMATORY",
              "source": "frozen DEV-018 original run 38802569971",
              "number_of_seed_regimes": 6, "by_regime": {}}
    for regime in REGIMES:
        d = {}
        for arm in ARMS:
            entries = [lookup[s, regime]["arms"][arm] for s in SEEDS]
            item = {"original_accuracy": sum(x["prefixes"]["512"]["accuracy"] for x in entries) / 3}
            for level in LEVELS:
                acc = [x[level]["heldout"]["accuracy"] for x in entries]
                shuffled = [x[level]["shuffled_label_control"]["accuracy"] for x in entries]
                item[level] = {"per_seed_heldout_accuracy": acc,
                               "mean_heldout_accuracy": sum(acc) / 3,
                               "per_seed_shuffled_accuracy": shuffled,
                               "passes_predeclared_diagnostic": all(
                                   a >= .75 and a - b >= .10 for a, b in zip(acc, shuffled))}
            item["mean_paired_prob_gap_by_prefix"] = {
                prefix: sum(x["prefixes"][prefix]["paired_mean_probability_gap"]
                            for x in entries) / 3 for prefix in PREFIXES}
            d[arm] = item
        result["by_regime"][regime] = d
    result["native_TAC_original_capability_claim"] = False
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("files", nargs="+", type=Path)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    originals = [json.loads(x.read_text()) for x in args.files]
    analysis = summarize(originals)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(analysis, indent=2) + "\n")
    print("DEV020_AUDITED " + json.dumps(analysis), flush=True)
