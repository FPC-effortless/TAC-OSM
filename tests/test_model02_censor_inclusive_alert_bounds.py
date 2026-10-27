"""Constructed examples for original first-eligible risk censoring, no science claim."""
import importlib.util
from datetime import date,timedelta
from pathlib import Path
import sys
import unittest

ROOT=Path(__file__).resolve().parents[1]/"specialist_models"
sys.path.insert(0,str(ROOT))
spec=importlib.util.spec_from_file_location("model02_censor_inclusive_alert_bounds",
                                            ROOT/"model02_censor_inclusive_alert_bounds.py")
m=importlib.util.module_from_spec(spec)
sys.modules[spec.name]=m
spec.loader.exec_module(m)
D=date(2013,11,1)
def records():
    return [
      (("A","M"),D,"CENSORED",.99,.01),
      (("B","M"),D,"POSITIVE",.80,.80),
      (("C","M"),D,"NEGATIVE",.70,.90),
      (("D","M"),D,"CENSORED",.60,.70),
      (("E","M"),D,"NEGATIVE",.20,.20),
      (("F","M"),D,"POSITIVE",.10,.10),
      (("G","M"),D,"NEGATIVE",.09,.09),
      (("H","M"),D,"NEGATIVE",.08,.08),
      (("I","M"),D,"NEGATIVE",.07,.07),
      (("J","M"),D,"CENSORED",.06,.06)
    ]

class CensorInclusiveContract(unittest.TestCase):
    def test_all_first_decision_outcome_denominators(self):
        r=m.bounded_alerts(records(),(.2,))
        self.assertEqual(r["first_eligible_distinct_drives"],10)
        self.assertEqual(r["first_decision_outcomes"]["POSITIVE"],2)
        self.assertEqual(r["first_decision_outcomes"]["CENSORED"],3)
        self.assertFalse(r["fitting_or_calibration_during_evaluation"])

    def test_top_k_bounds_keep_censored_unknown(self):
        r=m.bounded_alerts(records(),(.2,))["budgets"][0]
        a=r["models"]["eight_feature_logistic"]
        self.assertEqual(r["review_capacity_drives"],2)
        self.assertEqual((a["known_positive"],a["known_negative"],a["unknown_censored"]),(1,0,1))
        self.assertEqual(a["precision_lower_bound_if_unknown_negative"],.5)
        self.assertEqual(a["precision_upper_bound_if_unknown_positive"],1.0)
        self.assertEqual(a["false_alerts_lower_bound"],0)
        self.assertEqual(a["false_alerts_upper_bound"],1)

    def test_second_model_can_rank_differently(self):
        r=m.bounded_alerts(records(),(.2,))["budgets"][0]
        b=r["models"]["two_feature_lookup"]
        self.assertEqual((b["known_positive"],b["known_negative"],b["unknown_censored"]),(1,1,0))
        self.assertEqual(b["precision_lower_bound_if_unknown_negative"],.5)
        self.assertEqual(b["precision_upper_bound_if_unknown_positive"],.5)

    def test_reject_duplicate_physical_drives(self):
        with self.assertRaises(m.EvaluationError):
            m.bounded_alerts(records()+records()[:1])

    def test_reject_missing_outcome_as_fake_negative(self):
        rows=records()
        rows[0]=(rows[0][0],D,"SAFE",.99,.01)
        with self.assertRaises(m.EvaluationError):m.bounded_alerts(rows)

    def test_no_independent_claim_from_constructed_result(self):
        r=m.bounded_alerts(records(),(.2,))
        self.assertEqual(r["scientific_qualification"],"NOT_QUALIFIED")
        self.assertFalse(r["independent_evaluation"])
        self.assertFalse(r["prospective_prediction_time_visibility_proven"])

    def test_invalid_review_capacity(self):
        with self.assertRaises(m.EvaluationError):
            m.bounded_alerts(records(),(0.0,))
        with self.assertRaises(m.EvaluationError):
            m.bounded_alerts(records(),(2.0,))

    def test_empty_drive_list_denied(self):
        with self.assertRaises(m.EvaluationError):m.bounded_alerts([])

    def test_model_replay_identity_original(self):
        a,b=m.frozen.load_models()
        self.assertEqual(a["fit_examples"],112115)
        self.assertEqual(b["fit_windows"],112115)

if __name__=="__main__":
    unittest.main()
