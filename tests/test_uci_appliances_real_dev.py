"""Constructed fixtures for validating a real-data ingestion script; not skill evidence."""
import csv
from datetime import datetime,timedelta
import importlib.util
import io
import json
from pathlib import Path
import sys
import unittest
import zipfile

PATH=Path(__file__).resolve().parents[1]/"specialist_models"/"uci_appliances_real_dev.py"
spec=importlib.util.spec_from_file_location("uci_appliances_real_dev",PATH)
m=importlib.util.module_from_spec(spec)
sys.modules[spec.name]=m
spec.loader.exec_module(m)


def fake_zip(overrides=None, trim=None):
    out=io.StringIO()
    w=csv.DictWriter(out,fieldnames=["date","Appliances","lights"])
    w.writeheader()
    start=datetime(2016,1,1)
    for idx in range(m.DEV_END if trim is None else trim):
        d=(start+timedelta(minutes=10*idx)).strftime("%Y-%m-%d %H:%M:%S")
        y=70+20*(idx%144<40)+(idx%7)
        value=(overrides or {}).get(idx,y)
        w.writerow({"date":d,"Appliances":value,"lights":0})
    # Include a protected TEST row that is intentionally invalid: untouched.
    if trim is None:
        w.writerow({"date":"INVALID-PROTECTED-TEST-ROW","Appliances":"LEAK_SENTINEL","lights":0})
    buf=io.BytesIO()
    with zipfile.ZipFile(buf,"w",zipfile.ZIP_DEFLATED) as z:
        z.writestr("energydata_complete.csv",out.getvalue())
    return buf.getvalue()


class UCIRealSourceFixtureChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data=fake_zip()

    def test_valid_constructed_source_parses_without_test(self):
        x=m.original_rows(self.data)
        self.assertEqual(len(x),m.DEV_END)
        self.assertEqual(x[0],91)
    def test_score_separates_cal_and_dev(self):
        report=m.score_source(self.data)
        self.assertFalse(report["independent_test_scored"])
        self.assertFalse(report["trained_tac_model"])
        self.assertEqual(report["dev_cases"],m.DEV_END-m.CAL_END-m.H)
        self.assertEqual(report["calibration_cases"],m.CAL_END-m.FIT_END-m.H)
        self.assertIn(report["selected_using_cal_only"],
                      ("last_observation","same_time_previous_day"))
        self.assertTrue(0 <= report["selected_dev"]["coverage_90_observed"] <= 1)
    def test_protected_label_is_not_parsed_or_used(self):
        altered=fake_zip({m.CAL_END:10**5})
        baseline=m.score_source(self.data)
        changed=m.score_source(altered)
        self.assertNotEqual(changed["selected_dev"]["mae_wh"],
                            baseline["selected_dev"]["mae_wh"])
        # TEST sentinel remains invalid and never causes parsing failure.
        self.assertEqual(m.original_rows(altered)[m.CAL_END],10**5)
    def test_mutating_future_dev_cannot_change_cal_selection(self):
        original=m.score_source(self.data)
        modified=m.score_source(fake_zip({m.CAL_END+100:100000}))
        self.assertEqual(original["selected_using_cal_only"],modified["selected_using_cal_only"])
        self.assertEqual(original["calibration_mae_wh"],modified["calibration_mae_wh"])
        self.assertEqual(original["calibration_q_90_wh"],modified["calibration_q_90_wh"])
    def test_noncontiguous_timestamp_rejected(self):
        buf=io.StringIO();w=csv.writer(buf)
        w.writerow(["date","Appliances"])
        w.writerow(["2016-01-01 00:00:00","30"])
        w.writerow(["2016-01-01 00:30:00","31"])
        with io.BytesIO() as x:
            with zipfile.ZipFile(x,"w") as archive:
                archive.writestr("energydata_complete.csv",buf.getvalue())
            with self.assertRaises(m.SourceError):
                m.original_rows(x.getvalue())
    def test_tampered_source_target_rejected(self):
        with self.assertRaises(m.SourceError):
            m.original_rows(fake_zip({10:"FUTURE_LABEL"}))
    def test_unexpected_size_rejected(self):
        with self.assertRaises(m.SourceError):
            m.original_rows(fake_zip(trim=500))
    def test_quantile_explicit_rank(self):
        q,k=m.quantile(list(range(100)))
        self.assertEqual((q,k),(90,91))
    def test_protected_indices_disallowed(self):
        with self.assertRaises(m.SourceError):
            list(m.eligible_indices(m.CAL_END,m.DEV_END+1))
    def test_no_fake_selection_method(self):
        with self.assertRaises(m.SourceError):
            m.candidate_forecast([1,2,3],1,"future_cheat")
    def test_wrong_zip_rejected(self):
        with self.assertRaises(zipfile.BadZipFile):
            m.original_rows(b"not an official zip")

if __name__=="__main__":
    unittest.main()
