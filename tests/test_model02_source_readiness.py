"""Source receipt tamper and eligibility tests using committed historical evidence."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys
import unittest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location("model02_source_readiness",
                                            ROOT/"specialist_models"/"model02_source_readiness.py")
m=importlib.util.module_from_spec(spec)
sys.modules[spec.name]=m
spec.loader.exec_module(m)

class SourceReadinessTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixed=m.read(m.FIXED);cls.rolling=m.read(m.ROLLING);cls.original=m.read(m.ORIGINAL)

    def data(self):
        return deepcopy(self.fixed),deepcopy(self.rolling),deepcopy(self.original)

    def test_actual_receipts_block_training_without_invented_attestation(self):
        report=m.evaluate(*self.data())
        self.assertEqual(report["status"],"BLOCKED_FOR_QUALIFIED_TRAINING")
        self.assertFalse(report["qualified_fit_authorized"])
        self.assertFalse(report["training_performed"])
        self.assertIn("POST_EXPOSURE_ADMISSION_RULE_REQUIRES_NEW_SOURCE_COHORT",report["blockers"])
        self.assertIn("SOURCE_PREDICTION_TIME_AVAILABILITY_UNVERIFIED",report["blockers"])
        self.assertEqual(report["outcomes_by_phase"]["fit"]["distinct_positive_drives"],35)
        self.assertEqual(report["outcomes_by_phase"]["dev"]["distinct_positive_drives"],65)

    def test_reject_publisher_zip_digest_change(self):
        f,r,o=self.data();r["source_original_zip_sha256"]="0"*64
        with self.assertRaises(m.AuditError):m.evaluate(f,r,o)

    def test_reject_future_test_boundary_change(self):
        f,r,o=self.data();r["heldout_start_not_parsed"]="2013-11-30"
        with self.assertRaises(m.AuditError):m.evaluate(f,r,o)

    def test_reject_censor_denominator_inflation(self):
        f,r,o=self.data();r["fit_exposed"]["complete_observed_nonfailure_windows"]+=99999
        with self.assertRaises(m.AuditError):m.evaluate(f,r,o)

    def test_reject_event_count_inflation(self):
        f,r,o=self.data();r["fit_exposed"]["unique_drives_with_future_failure_windows"]=100000
        with self.assertRaises(m.AuditError):m.evaluate(f,r,o)

    def test_reject_negative_denominator(self):
        f,r,o=self.data();r["dev_exposed"]["right_censored_windows"]=-12
        with self.assertRaises(m.AuditError):m.evaluate(f,r,o)

    def test_reject_missing_original_per_day_hashes(self):
        f,r,o=self.data();o["daily"][0]["original_member_sha256"]="not-a-hash"
        with self.assertRaises(m.AuditError):m.evaluate(f,r,o)

    def test_claimed_signed_source_flag_cannot_remove_postexposure_gate(self):
        f,r,o=self.data()
        r["real_time_source_available_at_verified"]=True
        r["source_rights_and_original_failure_adjudication_verified"]=True
        r["independent_evaluation"]=True
        result=m.evaluate(f,r,o)
        self.assertFalse(result["qualified_fit_authorized"])
        self.assertIn("POST_EXPOSURE_ADMISSION_RULE_REQUIRES_NEW_SOURCE_COHORT",result["blockers"])

    def test_nonindependent_source_windows_distinct_from_drives(self):
        d=m.evaluate(*self.data())
        assert d["outcomes_by_phase"]["dev"]["positive_overlapping_windows"]==397
        self.assertNotEqual(d["outcomes_by_phase"]["dev"]["positive_overlapping_windows"],
                            d["outcomes_by_phase"]["dev"]["distinct_positive_drives"])

    def test_fail_when_exposure_study_label_rewritten(self):
        f,r,o=self.data();r["study"]="FRESH_BLIND_TEST"
        with self.assertRaises(m.AuditError):m.evaluate(f,r,o)

if __name__=="__main__":
    unittest.main()
