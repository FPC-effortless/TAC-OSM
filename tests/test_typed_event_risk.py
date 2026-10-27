"""Constructed-data unit tests. No genuine publisher/eval or commercial claims."""
import importlib.util
from datetime import datetime, timedelta, timezone
from pathlib import Path
import sys
import unittest

PATH=Path(__file__).resolve().parents[1]/"specialist_models"/"typed_event_risk.py"
spec=importlib.util.spec_from_file_location("typed_event_risk", PATH)
m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)

def fixture(day, signal, outcome=None):
    t=datetime(2026,1,1,tzinfo=timezone.utc)+timedelta(days=day)
    r={"observed_at":t.isoformat(),"features":{
       "signal":{"value":signal,"available_at":t.isoformat()}}}
    if outcome is not None:
        r["outcome"]=outcome;r["outcome_at"]=(t+timedelta(hours=2)).isoformat()
    return r

class ProductBaselineContractTests(unittest.TestCase):
    def setUp(self):
        self.source={"features":["signal"],
            "status":"source-claimed-not-independently-audited",
            "fit_cutoff":"2026-01-09T00:00:00Z",
            "source_url":"https://example.org/CONSTRUCTED-NOT-A-REAL-SOURCE",
            "dataset_license":"fixture only","data_steward":"fixture only"}
        self.rows=[fixture(i,float(i),int(i>=4)) for i in range(8)]
    def model(self):
        return m.fit(self.rows,self.source,"a"*64)
    def test_fit_infer_and_advisory(self):
        p=m.predict(self.model(),fixture(10,9))
        self.assertGreater(p["probability"],.5)
        self.assertEqual(p["authority"],"ADVISORY_ONLY")
        self.assertEqual(p["calibration"],"NOT_INDEPENDENTLY_QUALIFIED")
    def test_future_features_blocked(self):
        x=fixture(10,9)
        x["features"]["signal"]["available_at"]="2026-01-12T00:00:00Z"
        with self.assertRaises(m.EvidenceError):m.predict(self.model(),x)
    def test_labels_cannot_enter_inference(self):
        with self.assertRaises(m.EvidenceError):m.predict(self.model(),fixture(10,9,1))
    def test_fit_outcomes_must_exist_at_cutoff(self):
        with self.assertRaises(m.EvidenceError):
            m.fit(self.rows+[fixture(8,7,1)],self.source,"a"*64)
    def test_training_requires_both_classes(self):
        with self.assertRaises(m.EvidenceError):
            m.fit([fixture(i,i,1) for i in range(8)],self.source,"a"*64)
    def test_development_scoring_not_qualification(self):
        x=m.evaluate(self.model(),[fixture(10,0,0),fixture(11,10,1)])
        self.assertEqual(x["qualification"],"DEV_ONLY_NOT_INDEPENDENT_TEST")
        self.assertEqual(x["cases"],2)
    def test_development_cannot_overlap_fit_cutoff(self):
        with self.assertRaises(m.EvidenceError):
            m.evaluate(self.model(),[fixture(2,0,0)])
    def test_refuse_unqualified_source_metadata(self):
        self.source["source_url"]="made-up"
        with self.assertRaises(m.EvidenceError):self.model()
    def test_utc_required(self):
        with self.assertRaises(m.EvidenceError):
            m.utc("2026-01-02T00:00:00")
if __name__=="__main__":unittest.main()
