"""Constructed paired original-checkpoint evaluator tests (NOT source skill evidence)."""
from datetime import date,timedelta
import importlib.util
from pathlib import Path
import sys
import unittest

ROOT=Path(__file__).resolve().parents[1]/"specialist_models"
sys.path.insert(0,str(ROOT))
spec=importlib.util.spec_from_file_location("model02_paired_exposed_dev",ROOT/"model02_paired_exposed_dev.py")
m=importlib.util.module_from_spec(spec)
sys.modules[spec.name]=m
spec.loader.exec_module(m)

D=date(2013,11,1)
A=("TEST-A","MODEL-Z")
B=("TEST-B","MODEL-Y")
C=("TEST-C","MODEL-X")

def rows():
    # Reversed order for A: first decision cannot be selected by outcome.
    return [(A,D+timedelta(days=3),1,.9,.4),
            (A,D,0,.05,.15),
            (B,D,1,.8,.55),
            (B,D+timedelta(days=1),1,.7,.6),
            (C,D,0,.03,.07)]

class Model02PairedContractTests(unittest.TestCase):
    def test_earliest_per_physical_drive_not_positive_selected(self):
        subset=m.per_drive_first_matured(rows())
        self.assertEqual(len(subset),3)
        by_drive={v[0]:v for v in subset}
        self.assertEqual(by_drive[A][2],0)
        self.assertEqual(by_drive[A][1],D)

    def test_tied_case_brier_and_logloss_are_computed_on_same_inputs(self):
        sample=rows()
        lr=m.summarize(sample,3)
        lookup=m.summarize(sample,4)
        self.assertEqual(lr["n"],lookup["n"])
        self.assertEqual(lr["positive"],lookup["positive"])
        self.assertNotEqual(lr["brier"],lookup["brier"])

    def test_no_evidence_is_fabricated_for_empty_cases(self):
        self.assertEqual(m.summarize([],3)["status"],"NO_SCORED_CASES")

    def test_report_flags_no_fit_or_protected_scoring(self):
        logistic={"fit_positive_windows":1,"fit_examples":10}
        lookup={"prior_fit":.1}
        sample=rows()
        report=m.pair_report(sample,{"matured_windows":len(sample)},logistic,lookup)
        self.assertFalse(report["fit_performed_during_comparison"])
        self.assertFalse(report["independent_evaluation"])
        self.assertFalse(report["protected_future_after_2013_11_15_parsed"])
        self.assertFalse(report["architecture_superiority_claim"])
        self.assertEqual(report["paired_results"]["first_matured_forecast_per_drive"]
                         ["eight_feature_logistic"]["n"],3)

    def test_frozen_input_denominator_inconsistency_rejected(self):
        with self.assertRaises(m.PairError):
            m.pair_report(rows(),{"matured_windows":400},{},{})

    def test_original_checkpoint_both_recoverable(self):
        logistic,lookup=m.load_models()
        self.assertEqual(logistic["model_type"],"CLASSICAL_LOGISTIC_NOT_TAC")
        self.assertEqual(lookup["fit_windows"],logistic["fit_examples"])
        self.assertEqual(len(logistic["coefficients"]),8)

    def test_posthoc_uncertainty_is_deterministic_not_confirmation(self):
        first=m.per_drive_first_matured(rows())
        a=m.first_drive_bootstrap_posthoc(first,replicates=60,seed=3)
        b=m.first_drive_bootstrap_posthoc(first,replicates=60,seed=3)
        self.assertEqual(a,b)
        self.assertIn("POST_HOC",a["status"])
        self.assertEqual(a["positive_decisions"],1)
        self.assertEqual(a["sampled_drive_count"],3)
        self.assertEqual(len(a["percentile_95_bounds"]),2)

    def test_first_per_drive_representative_is_sorted_temporally(self):
        first=m.per_drive_first_matured(list(reversed(rows())))
        self.assertEqual({x[0]:x[1] for x in first}[A],D)

    def test_all_source_keys_are_internal_aggregation_not_exported(self):
        report=m.pair_report(rows(),{"matured_windows":len(rows())},
                             {"fit_positive_windows":1,"fit_examples":10},
                             {"prior_fit":.1})
        value=m.canonical(report)
        self.assertNotIn("TEST-A",value)
        self.assertNotIn("TEST-B",value)
        self.assertNotIn("TEST-C",value)

if __name__=="__main__":
    unittest.main()
