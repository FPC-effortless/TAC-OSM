"""Constructed physical-drive clustered score diagnostics, not new real ability evidence."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from datetime import timedelta
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]/"specialist_models"
sys.path.insert(0,str(ROOT))
spec=importlib.util.spec_from_file_location("model02_drive_cluster_sensitivity",
                                            ROOT/"model02_drive_cluster_sensitivity.py")
m=importlib.util.module_from_spec(spec)
sys.modules[spec.name]=m
spec.loader.exec_module(m)


def source_fixture():
    first=m.exp.period.FIT_END-timedelta(days=9)
    start=m.exp.period.FIT_END
    end=m.exp.period.EXPOSED_END
    a={}
    b={}
    for i in range((end-first).days):
        d=first+timedelta(days=i)
        if d<start+timedelta(days=10):
            a[d]={"failed":d==start+timedelta(days=9),"cell":"POSITIVE|ZERO"}
        b[d]={"failed":False,"cell":"ZERO|ZERO"}
    return {("testA","modelA"):a,("testB","modelB"):b}

def fixed_model():
    cells={}
    for x in ("MISSING","ZERO","POSITIVE"):
        for y in ("MISSING","ZERO","POSITIVE"):
            cells[x+"|"+y]={"p":.1 if x=="POSITIVE" else .01,"n":1,"pos":0}
    return {"prior_fit":.03,"cells":cells,"schema":"model02-classical-smart-binary-original-v1"}

class DriveClusterTests(unittest.TestCase):
    def calc(self,mutate=False):
        timelines=source_fixture()
        counts,denom=m.exp.case_rows(timelines,m.exp.period.FIT_END,m.exp.period.EXPOSED_END)
        receipt={"DEV":{"eligible_windows":denom["eligible_windows"],
                        "right_censored_windows":denom["right_censored_windows"],
                        "matured_windows":denom["matured_windows"],
                        "positive_overlapping_windows":sum(p for p,n in counts.values()),
                        "distinct_positive_drives":denom["distinct_drives_with_positive_windows"],
                        "brier":.5,
                        "fit_only_prior_baseline_brier":.5,
                        "log_loss":.5,
                        "fit_only_prior_baseline_log_loss":.5}}
        if mutate:receipt["DEV"]["matured_windows"]+=1
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"receipt.json"
            path.write_text(json.dumps(receipt))
            with patch.object(m,"RESULT",path):
                return m.aggregate(fixed_model(),timelines)

    def test_macro_uses_unique_drive_not_observation_windows(self):
        x=self.calc()
        self.assertEqual(x["distinct_dev_drives_with_matured_windows"],2)
        self.assertEqual(x["equal_drive_weighted"]["lookup_brier"]["n_distinct_drives"],2)
        self.assertGreater(x["original_exposed_dev_matured_windows"],2)

    def test_first_matured_window_one_per_drive(self):
        x=self.calc()
        self.assertEqual(x["one_earliest_matured_window_per_drive"]["lookup_brier"]["n_distinct_drives"],2)
        self.assertFalse(x["independent_evaluation"])
        self.assertFalse(x["fit_performed_this_run"])

    def test_physical_entity_failure_not_independent_windows(self):
        x=self.calc()
        self.assertEqual(x["distinct_dev_drives_with_positive_windows"],1)
        self.assertGreater(x["original_exposed_dev_positive_windows"],1)

    def test_tampered_original_denominator_blocked(self):
        with self.assertRaises(m.exp.EvidenceError):self.calc(mutate=True)

    def test_no_identifying_drive_ids_emitted(self):
        s=json.dumps(self.calc())
        self.assertNotIn("testA",s)
        self.assertNotIn("testB",s)

    def test_proper_probability_scoring(self):
        b,l=m.measures(.2,1)
        self.assertAlmostEqual(b,.64)
        self.assertAlmostEqual(l,-__import__("math").log(.2))

    def test_empty_source_drive_group_fails(self):
        with self.assertRaises(m.exp.EvidenceError):m.summarize([])

    def test_source_status_still_unqualified(self):
        x=self.calc()
        self.assertEqual(x["scientific_qualification"],"NOT_QUALIFIED")
        self.assertFalse(x["protected_future_member_contents_parsed"])

if __name__=="__main__":
    unittest.main()
