"""Constructed unit-cost algebra, plus pinned previously scored real source receipt."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys
import unittest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location("model02_review_cost_envelope",
                                            ROOT/"specialist_models"/"model02_review_cost_envelope.py")
m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)

class CostEnvelope(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original=json.loads(m.SOURCE.read_bytes())
    def test_exact_original_case_cost_thresholds(self):
        r=m.compute(self.original)
        self.assertFalse(r["actual_business_costs_measured"])
        self.assertFalse(r["qualified_operator_recommendation"])
        self.assertFalse(r["source_has_verified_prevention_rate"])
        one=r["review_budgets"][0]["models"]
        self.assertAlmostEqual(
            one["eight_feature_logistic"]["benefit_per_prevented_event_over_review_cost_break_even_if_all_unknown_negative"],263/16)
        self.assertAlmostEqual(
            one["eight_feature_logistic"]["benefit_per_prevented_event_over_review_cost_break_even_best_case"],263/17)
        self.assertAlmostEqual(
            one["two_feature_lookup"]["benefit_per_prevented_event_over_review_cost_break_even_if_all_unknown_negative"],263/10)
    def test_negative_unknown_stays_unknown(self):
        out=m.compute(self.original)
        for b in out["review_budgets"]:
            for candidate in b["models"].values():
                self.assertEqual(candidate["economic_net_benefit"],"NOT_MEASURABLE")
                self.assertEqual(candidate["actual_prevented_failures"],"UNKNOWN")
    def test_larger_capacity_requires_more_review_work(self):
        r=m.compute(self.original)
        self.assertEqual(r["review_budgets"][0]["reviewed_drives"],263)
        self.assertEqual(r["review_budgets"][1]["reviewed_drives"],1315)
        self.assertGreater(r["review_budgets"][1]["models"]["eight_feature_logistic"]
                           ["benefit_per_prevented_event_over_review_cost_break_even_if_all_unknown_negative"],
                           r["review_budgets"][0]["models"]["eight_feature_logistic"]
                           ["benefit_per_prevented_event_over_review_cost_break_even_if_all_unknown_negative"])
    def test_manipulated_original_source_hash_blocks(self):
        r=deepcopy(self.original);r["source_zip_sha256"]="0"*64
        with self.assertRaises(m.EconomicError):m.compute(r)
    def test_manipulated_outcome_count_blocks(self):
        r=deepcopy(self.original);r["first_decision_outcomes"]["POSITIVE"]=300
        with self.assertRaises(m.EconomicError):m.compute(r)
    def test_corrupt_per_model_alert_partition_blocks(self):
        r=deepcopy(self.original)
        r["budgets"][0]["models"]["eight_feature_logistic"]["unknown_censored"]=0
        with self.assertRaises(m.EconomicError):m.compute(r)
    def test_qualified_status_is_not_derived_from_high_possible_capture(self):
        r=deepcopy(self.original)
        r["independent_evaluation"]=True
        with self.assertRaises(m.EconomicError):m.compute(r)
    def test_denominator_and_first_source_match_expected(self):
        r=m.compute(self.original)
        self.assertEqual(r["original_first_eligible_drive_count"],26311)
        self.assertEqual(len(r["review_budgets"]),2)
if __name__=="__main__":
    unittest.main()
