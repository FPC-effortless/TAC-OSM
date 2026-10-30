"""Constructed analysis checks, never trained capability evidence."""
import copy
import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "research/mathematical-engineering/langacquire097/analysis.py"
spec = importlib.util.spec_from_file_location("langacquire097_analysis_test", PATH)
analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)


def fixture(improvement=True):
    cases = []
    for domain in ("lexical", "arc"):
        for index in range(4):
            cases.append(dict(id=f"{domain}-{index}", domain=domain, partition="DEV",
                              metadata=dict(cluster_id=f"{domain}-source-{index}",
                                            group="positive" if domain == "lexical" else "arc",
                                            binding_shuffle_available=domain == "arc")))
    rows = []
    for objective in analysis.OBJECTIVES:
        for seed in analysis.SEEDS:
            for condition in analysis.CONDITIONS:
                for case in cases:
                    good = improvement and objective == "query" and condition == "carry"
                    rows.append(dict(family="tac", objective=objective, seed=seed,
                        condition=condition, domain=case["domain"], case_id=case["id"],
                        cluster_id=case["metadata"]["cluster_id"], group=case["metadata"]["group"],
                        nll=1. if good else 2., exact_match=good, format_valid=True))
    integrity = dict.fromkeys(analysis.REQUIRED_INTEGRITY, "PASS")
    return cases, rows, integrity


def binding_rows(rows):
    return [{**row, "condition": "binding_shuffle", "nll": 2., "exact_match": False}
            for row in rows if row["domain"] == "arc" and row["condition"] == "carry"]


class Analysis097Tests(unittest.TestCase):
    def run_fixture(self, improvement=True):
        cases, rows, integrity = fixture(improvement)
        return analysis.analyze(rows, cases, integrity, families=("tac",), binding_rows=binding_rows(rows))

    def test_null_effects_fail_every_capability_gate(self):
        report = self.run_fixture(False)
        self.assertEqual(report["status"], "COMPLETED_INTERNAL_MEASUREMENT")
        self.assertEqual(report["families"][0]["joint_learning_gate"], "FAIL")
        self.assertEqual(report["cross_architecture_superiority"], "INELIGIBLE")
        self.assertFalse(report["general_intelligence_established"])

    def test_paired_improvements_do_not_create_architecture_or_agi_claim(self):
        report = self.run_fixture()
        item = report["families"][0]
        self.assertEqual(item["joint_learning_gate"], "PASS")
        effect = item["domains"]["lexical"]["effects"]["query_minus_full_nll"]
        self.assertEqual(effect["cases"], 4)
        self.assertEqual(effect["clusters"], 4)
        self.assertEqual(effect["ci95_lower"], 1.)
        self.assertEqual(effect["positive_seeds"], 5)
        self.assertEqual(report["cross_architecture_superiority"], "INELIGIBLE")

    def test_missing_prediction_is_preserved_as_inconclusive(self):
        cases, rows, integrity = fixture()
        result = analysis.analyze(rows[:-1], cases, integrity, families=("tac",))
        self.assertEqual(result["status"], "INCONCLUSIVE")
        self.assertEqual(result["expected_prediction_rows"] - result["prediction_rows"], 1)

    def test_duplicate_case_wrong_metadata_and_nonfinite_scores_fail(self):
        cases, rows, integrity = fixture()
        changes = [rows + [rows[0]],
                   [{**rows[0], "cluster_id": "wrong"}, *rows[1:]],
                   [{**rows[0], "nll": float("nan")}, *rows[1:]],
                   [{**rows[0], "nll": True}, *rows[1:]],
                   [{**rows[0], "exact_match": 1, "format_valid": 0}, *rows[1:]]]
        for changed in changes:
            with self.subTest(changed=changed[0]):
                self.assertEqual(analysis.analyze(changed, cases, integrity,
                                                  families=("tac",))["status"], "FAIL")

    def test_integrity_unknown_cannot_become_positive_finding(self):
        cases, rows, integrity = fixture()
        del integrity["gradient_to_acquisition"]
        result = analysis.analyze(rows, cases, integrity, families=("tac",))
        self.assertEqual(result["status"], "INCONCLUSIVE")
        self.assertEqual(result["families"][0]["joint_learning_gate"], "INCONCLUSIVE")

    def test_concrete_integrity_violation_is_failure_not_unknown(self):
        cases, rows, integrity = fixture()
        integrity["prediction_boundary"] = "FAIL"
        report = analysis.analyze(rows, cases, integrity, families=("tac",))
        self.assertEqual(report["status"], "FAIL")
        self.assertEqual(report["integrity_status"], "FAIL")
        self.assertFalse(report["families"])

    def test_raw_hurdle_is_separate_and_reference_coverage_is_complete(self):
        cases, rows, integrity = fixture()
        raw = [dict(case_id=case["id"], nll=.5, exact_match=True)
               for case in cases if case["domain"] == "lexical"]
        result = analysis.analyze(rows, cases, integrity, families=("tac",), raw_rows=raw,
                                  binding_rows=binding_rows(rows))
        self.assertEqual(result["families"][0]["joint_learning_gate"], "PASS")
        self.assertEqual(result["families"][0]["raw_history_reference"]["quality_hurdle"], "FAIL")
        self.assertEqual(analysis.analyze(rows, cases, integrity, families=("tac",),
                                        raw_rows=raw[:-1])["status"], "FAIL")

    def test_unavailable_donor_is_preserved_without_replacing_the_case(self):
        cases, rows, integrity = fixture()
        cases[4]["metadata"]["donor_available"] = False
        for row in rows:
            if row["case_id"] == cases[4]["id"] and row["condition"] == "donor":
                row.update(status="INCONCLUSIVE", nll=None, exact_match=None, format_valid=None)
        result = analysis.analyze(rows, cases, integrity, families=("tac",), binding_rows=binding_rows(rows))
        item = result["families"][0]
        effect = item["domains"]["arc"]["effects"]["carry_minus_donor_exact"]
        self.assertEqual(effect["status"], "INCONCLUSIVE")
        self.assertEqual(effect["available_cases"], 3)
        self.assertEqual(effect["unavailable_cases"], 1)
        self.assertEqual(item["joint_learning_gate"], "PASS")

    def test_unavailable_binding_control_blocks_positive_structural_attribution(self):
        cases, rows, integrity = fixture()
        cases[4]["metadata"]["binding_shuffle_available"] = False
        bindings = binding_rows(rows)
        for row in bindings:
            if row["case_id"] == cases[4]["id"]:
                row.update(status="INCONCLUSIVE", nll=None, exact_match=None, format_valid=None)
        result = analysis.analyze(rows, cases, integrity, families=("tac",), binding_rows=bindings)
        item = result["families"][0]
        self.assertEqual(item["domains"]["arc"]["gates"]["unknown_rule_exact_learning"], "PASS")
        self.assertEqual(item["domains"]["arc"]["gates"]["binding_specific_attribution"], "INCONCLUSIVE")
        self.assertEqual(item["joint_learning_gate"], "INCONCLUSIVE")

    def test_both_external_reference_hurdles_remain_distinct_from_learning(self):
        cases, rows, integrity = fixture()
        raw = [dict(case_id=case["id"], nll=2., exact_match=False)
               for case in cases if case["domain"] == "lexical"]
        arc = [dict(case_id=case["id"], exact_match=False)
               for case in cases if case["domain"] == "arc"]
        report = analysis.analyze(rows, cases, integrity, families=("tac",),
            raw_rows=raw, binding_rows=binding_rows(rows), arc_reference_rows=arc)
        self.assertEqual(report["families"][0]["practical_reference_quality_gate"], "PASS")
        for row in arc:
            row["exact_match"] = True
        report = analysis.analyze(rows, cases, integrity, families=("tac",),
            raw_rows=raw, binding_rows=binding_rows(rows), arc_reference_rows=arc)
        self.assertEqual(report["families"][0]["joint_learning_gate"], "PASS")
        self.assertEqual(report["families"][0]["practical_reference_quality_gate"], "FAIL")

    def test_frozen_judge_mapping_is_supported_without_actor_metadata(self):
        cases, rows, integrity = fixture()
        judge = {case["id"]: dict(domain=case["domain"], **case["metadata"])
                 for case in cases}
        report = analysis.analyze(rows, judge, integrity, families=("tac",),
                                  binding_rows=binding_rows(rows))
        self.assertEqual(report["status"], "COMPLETED_INTERNAL_MEASUREMENT")
        arc_summary = [row for row in report["families"][0]["summary"] if row["domain"] == "arc"]
        self.assertEqual(len(arc_summary), 8)
        self.assertTrue(all(row["group"] == "unknown_rule" for row in arc_summary))

    def test_seed_copies_are_not_resampled_as_independent_cases(self):
        matrix = np.tile(np.array([1., -1., 1., -1.]), (5, 1))
        effect = analysis._effect(matrix, ["a", "a", "b", "b"])
        self.assertEqual(effect["cases"], 4)
        self.assertEqual(effect["clusters"], 2)
        self.assertEqual(effect["ci95_lower"], 0.)
        self.assertEqual(effect["ci95_upper"], 0.)
        self.assertEqual(effect["positive_clusters"], 0)

    def test_row_order_and_file_preservation(self):
        cases, rows, integrity = fixture()
        first = analysis.analyze(rows, cases, integrity, families=("tac",))
        second = analysis.analyze(list(reversed(rows)), list(reversed(cases)),
                                  integrity, families=("tac",))
        self.assertEqual(first, second)
        with tempfile.TemporaryDirectory() as folder:
            json_path, markdown_path = Path(folder) / "results.json", Path(folder) / "results.md"
            analysis.write_reports(first, json_path, markdown_path)
            with self.assertRaises(FileExistsError):
                analysis.write_reports(second, json_path, markdown_path)


if __name__ == "__main__":
    unittest.main()
