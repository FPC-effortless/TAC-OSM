"""Constructed evaluator tests, never real-data persistence evidence."""
import copy
import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("langpersist096_analysis", ROOT /
        "research/mathematical-engineering/langpersist096/analysis.py")
analysis = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(analysis)


def distribution(probabilities, count=300):
    result = np.empty((len(probabilities), count, 256), dtype=np.float32)
    for condition, probability in enumerate(probabilities):
        result[condition] = np.log((1 - probability) / 255)
        result[condition, :, 42] = np.log(probability)
    return result


class Analysis096Tests(unittest.TestCase):
    def fixture(self, probabilities=(.2, .1, .08, .05)):
        records = []
        for publisher_index, publisher in enumerate(analysis.PUBLISHERS):
            for group in analysis.GROUPS:
                for _ in range(20):
                    records.append({"publisher": publisher, "group": group,
                        "file": str(20 + publisher_index * 2) + ".bin",
                        "target_offset": 192 + len(records) * 256})
        targets = np.tile(np.array([42, 43, 44, 45], dtype=np.uint8), (300, 1))
        greedy = np.zeros((4, 300, 4), dtype=np.uint8)
        greedy[:, :, 0] = 42
        greedy[0] = targets
        prediction = {"log_probs": distribution(probabilities), "greedy": greedy}
        predictions = {key: prediction for key in analysis.REQUIRED_MODELS
                       if key[0] != "raw_history"}
        predictions[("raw_history", 0)] = {
            "log_probs": distribution((.1,)), "greedy": greedy[1:2]}
        costs = {key: {"setup_seconds": .1, "acquisition_seconds": .2,
                "query_seconds": .3, "generation_seconds": .4,
                "model_bytes": 2_000_000, "state_payload_bytes": 3904 * 300,
                "serialized_metadata_bytes": 512 * 300, "peak_rss_bytes": 20_000_000}
                for key in predictions}
        for key, row in costs.items():
            if key[0] != "raw_history":
                row["per_condition_costs"] = {condition: {
                    "intervention_seconds": .01, "query_seconds": .065,
                    "generation_seconds": .1} for condition in analysis.CONDITIONS}
        return predictions, targets, records, costs

    def test_complete_paired_positive_result_and_accounting(self):
        args = self.fixture()
        result = analysis.analyze(*args)
        self.assertEqual(result["status"], "POSITIVE_INTERNAL_FINDING")
        self.assertEqual(result["primary"]["nll_gate"]["status"], "PASS")
        self.assertEqual(result["primary"]["exact4_utility_gate"]["status"], "PASS")
        effect = result["primary"]["effects"]["reset_minus_carry_nll"]
        self.assertAlmostEqual(effect["mean"], np.log(2), places=6)
        self.assertEqual(result["primary"]["positive_publishers"], 5)
        self.assertEqual(result["primary"]["positive_seeds"], 3)
        self.assertEqual(result["resource_accounting"]["status"], "PASS")
        self.assertAlmostEqual(result["resource_accounting"]["total_recorded_worker_seconds"], 10)
        self.assertEqual(len(result["arm_means"]), 3 * 4 * 3 * 6)
        self.assertEqual(result["resources"][0]["state_payload_bytes_per_episode"], 3904)
        self.assertAlmostEqual(result["resources"][0]["per_condition_costs"]["carry"]
            ["with_shared_setup_acquisition_seconds"], .475)
        self.assertIn("UNMATCHED", result["architecture_superiority"])
        self.assertEqual(result["full_intelligence_theory"], "OPEN")

    def test_negative_prediction_result_cannot_pass_primary_gate(self):
        result = analysis.analyze(*self.fixture((.1, .2, .3, .4)))
        self.assertEqual(result["status"], "NEGATIVE_RESULT")
        self.assertEqual(result["primary"]["nll_gate"]["status"], "FAIL")
        self.assertFalse(result["primary"]["nll_gate"]["criteria"]["nll_ci95_lower_above_zero"])

    def test_donor_gate_blocks_non_content_specific_effect(self):
        result = analysis.analyze(*self.fixture((.2, .1, .3, .05)))
        self.assertEqual(result["status"], "NEGATIVE_RESULT")
        self.assertFalse(result["primary"]["nll_gate"]["criteria"]["donor_nll_ci95_lower_above_zero"])

    def test_exact_four_gate_stays_failed_when_nll_passes(self):
        predictions, targets, records, costs = self.fixture()
        predictions[("tac", 1701)]["greedy"][0, :, 1:] = 0
        result = analysis.analyze(predictions, targets, records, costs)
        self.assertEqual(result["primary"]["nll_gate"]["status"], "PASS")
        self.assertEqual(result["primary"]["exact4_utility_gate"]["status"], "FAIL")

    def test_nonfinite_or_unnormalized_distribution_fails(self):
        for corruption in (float("nan"), float("-inf"), 1.):
            with self.subTest(corruption=corruption):
                predictions, targets, records, costs = self.fixture()
                predictions[("tac", 1701)]["log_probs"][0, 0, 42] = corruption
                result = analysis.analyze(predictions, targets, records, costs)
                self.assertEqual(result["status"], "FAIL")

    def test_prediction_byte_inconsistency_and_wrong_shape_fail(self):
        predictions, targets, records, costs = self.fixture()
        predictions[("tac", 1701)]["greedy"][0, 0, 0] = 17
        self.assertEqual(analysis.analyze(predictions, targets, records, costs)["status"], "FAIL")
        predictions, targets, records, costs = self.fixture()
        predictions[("tac", 1701)] = {"log_probs": np.zeros((3, 300, 256)),
                                      "greedy": np.zeros((3, 300, 4), dtype=np.uint8)}
        self.assertEqual(analysis.analyze(predictions, targets, records, costs)["status"], "FAIL")

    def test_missing_checkpoint_or_raw_control_is_inconclusive(self):
        for missing in (("tac", 1703), ("raw_history", 0)):
            with self.subTest(missing=missing):
                predictions, targets, records, costs = self.fixture()
                predictions.pop(missing)
                result = analysis.analyze(predictions, targets, records, costs)
                self.assertEqual(result["status"], "INCONCLUSIVE")
                self.assertIn({"arm": missing[0], "seed": missing[1]}, result["missing_models"])
                if missing[0] == "tac":
                    self.assertEqual(result["primary"]["nll_gate"]["status"], "INCONCLUSIVE")

    def test_underfilled_cohort_is_not_relaxed_or_silently_dropped(self):
        predictions, targets, records, costs = self.fixture()
        predictions = {key: {name: array[:, 1:] for name, array in value.items()}
                       for key, value in predictions.items()}
        result = analysis.analyze(predictions, targets[1:], records[1:], costs)
        self.assertEqual(result["status"], "INSUFFICIENT_COHORT")
        self.assertEqual(result["cohort"]["counts"][analysis.PUBLISHERS[0]]["positive"], 19)
        self.assertEqual(result["primary"]["nll_gate"]["status"], "INCONCLUSIVE")

    def test_unknown_group_duplicate_offset_and_wrong_target_fail(self):
        for problem in ("group", "duplicate", "target"):
            with self.subTest(problem=problem):
                predictions, targets, records, costs = self.fixture()
                if problem == "group":
                    records[0]["group"] = "selected_after_scoring"
                elif problem == "duplicate":
                    records[1]["target_offset"] = records[0]["target_offset"]
                else:
                    targets = targets.astype(np.int64)
                    targets[0, 0] = 256
                self.assertEqual(analysis.analyze(predictions, targets, records, costs)["status"], "FAIL")

    def test_missing_cost_is_unknown_and_invalid_cost_fails(self):
        predictions, targets, records, costs = self.fixture()
        costs[("tac", 1701)]["acquisition_seconds"] = None
        result = analysis.analyze(predictions, targets, records, costs)
        self.assertEqual(result["status"], "INCONCLUSIVE")
        self.assertIsNone(result["resource_accounting"]["total_recorded_worker_seconds"])
        self.assertEqual(result["primary"]["nll_gate"]["status"], "PASS")
        costs[("tac", 1701)]["acquisition_seconds"] = float("nan")
        self.assertEqual(analysis.analyze(predictions, targets, records, costs)["status"], "FAIL")

    def test_cluster_bootstrap_never_treats_bytes_or_seeds_as_clusters(self):
        cluster_values = np.array([.1, .2, .3, .4, .9])
        publishers = np.repeat(analysis.PUBLISHERS, 20)
        values = np.repeat(cluster_values, 20)
        draws = np.random.default_rng(96001).integers(0, 5, (5000, 5))
        actual = analysis._cluster_interval(values, publishers, draws)
        expected = np.quantile(cluster_values[draws].mean(1), (.025, .975))
        np.testing.assert_allclose([actual["ci95_lower"], actual["ci95_upper"]], expected)
        repeated = analysis._cluster_interval(np.repeat(values, 3),
                                              np.repeat(publishers, 3), draws)
        np.testing.assert_allclose([repeated["ci95_lower"], repeated["ci95_upper"]], expected)

    def test_positive_negative_interaction_is_descriptive_and_paired(self):
        predictions, targets, records, costs = self.fixture()
        negative = np.array([record["group"] == "negative" for record in records])
        replacement = distribution((.2, .2, .08, .05))
        predictions[("tac", 1701)]["log_probs"][:, negative] = replacement[:, negative]
        result = analysis.analyze(predictions, targets, records, costs)
        interaction = result["primary"]["positive_minus_negative_nll_interaction"]
        self.assertEqual(interaction["status"], "DESCRIPTIVE_ONLY")
        self.assertFalse(interaction["used_for_gating_or_selection"])
        self.assertAlmostEqual(interaction["effect"]["mean"], np.log(2), places=6)
        self.assertEqual(result["primary"]["nll_gate"]["status"], "PASS")

    def test_condition_costs_cannot_disappear_or_disagree_with_flat_totals(self):
        predictions, targets, records, costs = self.fixture()
        costs[("tac", 1701)].pop("per_condition_costs")
        result = analysis.analyze(predictions, targets, records, costs)
        self.assertEqual(result["resource_accounting"]["status"], "INCONCLUSIVE")
        self.assertIsNone(result["resources"][0]["per_condition_costs"]["carry"]
                          ["with_shared_setup_acquisition_seconds"])
        predictions, targets, records, costs = self.fixture()
        costs[("tac", 1701)]["per_condition_costs"]["carry"]["query_seconds"] += .1
        self.assertEqual(analysis.analyze(predictions, targets, records, costs)["status"], "FAIL")

    def test_raw_single_condition_costs_have_no_invented_intervention(self):
        result = analysis.analyze(*self.fixture())
        raw = next(row for row in result["resources"] if row["arm"] == "raw_history")
        self.assertEqual(raw["conditions_measured"], 1)
        self.assertEqual(raw["per_condition_costs"]["carry"]["intervention_seconds"], 0)
        self.assertIn("not applicable", raw["per_condition_costs"]["carry"]["intervention_basis"])
        self.assertAlmostEqual(raw["recorded_total_seconds"],
            raw["per_condition_costs"]["carry"]["with_shared_setup_acquisition_seconds"])

    def test_json_markdown_are_reviewable_and_exclusive_create(self):
        result = analysis.analyze(*self.fixture())
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            json_path, markdown_path = root / "result.json", root / "result.md"
            analysis.write_reports(result, json_path, markdown_path)
            self.assertIn("POSITIVE_INTERNAL_FINDING", json_path.read_text())
            self.assertIn("NLL gate: **PASS**", markdown_path.read_text())
            self.assertIn("remains OPEN", markdown_path.read_text())
            original = json_path.read_bytes()
            with self.assertRaises(FileExistsError):
                analysis.write_reports(result, json_path, markdown_path)
            self.assertEqual(original, json_path.read_bytes())


if __name__ == "__main__":
    unittest.main()
