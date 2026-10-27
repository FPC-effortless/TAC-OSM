"""Constructed source-record and risk lookup tests; actual model uses real source separately."""
import csv
from datetime import timedelta
import io
import importlib.util
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
import zipfile

ROOT=Path(__file__).resolve().parents[1]/"specialist_models"
sys.path.insert(0,str(ROOT))
spec=importlib.util.spec_from_file_location("model02_backblaze_exploratory",
                                            ROOT/"model02_backblaze_exploratory.py")
m=importlib.util.module_from_spec(spec)
sys.modules[spec.name]=m
spec.loader.exec_module(m)

def fake_zip(*,nov_failure=True,invalid_hidden=True,missing_day=None):
    buf=io.BytesIO()
    with zipfile.ZipFile(buf,"w",zipfile.ZIP_DEFLATED) as z:
        for i,d in enumerate(m.period.days()):
            if str(d)==missing_day:continue
            f=io.StringIO()
            w=csv.writer(f)
            w.writerow(["date","serial_number","model","failure","smart_5_raw","smart_197_raw"])
            if i<=12:
                w.writerow([str(d),"A","A-MODEL",int(i==12),"3" if i>8 else "0","0"])
            w.writerow([str(d),"B","B-MODEL",0,"0","0"])
            if i>=23 and (not nov_failure or i<=41):
                w.writerow([str(d),"C","C-MODEL",int(nov_failure and i==41),"2",""])
            z.writestr("publisher/"+str(d)+".csv",f.getvalue())
        if invalid_hidden:z.writestr("publisher/2013-11-16.csv","not even a CSV")
    return buf.getvalue()

class OriginalRiskFixtureTests(unittest.TestCase):
    def source(self,**args):
        payload=fake_zip(**args)
        with patch.object(m,"ORIGINAL_SHA",m.pub.sha(payload)):
            return m.records_from_original(payload)
    def test_asof_causal_source_day_and_future_protected_member_not_read(self):
        rows=self.source()
        self.assertEqual(len(rows),3)
        self.assertIn(m.period.FIRST,rows[("A","A-MODEL")])
        self.assertEqual(rows[("B","B-MODEL")][m.period.FIRST]["cell"],"ZERO|ZERO")
        self.assertEqual(rows[("C","C-MODEL")][m.period.FIRST+timedelta(days=25)]["cell"],"POSITIVE|MISSING")
    def test_fit_does_not_use_nov_failure(self):
        a=self.source(nov_failure=True)
        b=self.source(nov_failure=False)
        ca,sa=m.case_rows(a,m.period.FIRST,m.period.FIT_END)
        cb,sb=m.case_rows(b,m.period.FIRST,m.period.FIT_END)
        self.assertEqual(dict(ca),dict(cb))
        self.assertEqual(sa,sb)
    def test_maturation_and_right_censoring(self):
        timeline=self.source()
        fit,status=m.case_rows(timeline,m.period.FIRST,m.period.FIT_END)
        self.assertTrue(status["right_censored_windows"]>0)
        self.assertEqual(status["distinct_drives_with_positive_windows"],1)
        self.assertEqual(status["matured_windows"],sum(v[1] for v in fit.values()))
    def test_meaningful_classical_reference_probability(self):
        counts={"ZERO|ZERO":[0,100],"POSITIVE|ZERO":[8,20]}
        model=m.fit(counts)
        self.assertTrue(m.risk(model,"POSITIVE|ZERO")>m.risk(model,"ZERO|ZERO"))
        report=m.score(model,counts,1)
        self.assertEqual(report["matured_windows"],120)
        self.assertEqual(report["positive_overlapping_windows"],8)
        self.assertEqual(model["scientific_qualification"],"EXPLORATORY_POST_EXPOSURE_NOT_QUALIFIED")
    def test_unobserved_feature_uses_missing_bin(self):
        self.assertEqual(m.bin_value(None),"MISSING")
        self.assertEqual(m.bin_value("0"),"ZERO")
        self.assertEqual(m.bin_value("1"),"POSITIVE")
        with self.assertRaises(m.EvidenceError):m.bin_value("-1")
    def test_source_zip_hash_is_binding(self):
        with self.assertRaises(m.EvidenceError):m.records_from_original(fake_zip())
    def test_missing_source_day_fails(self):
        data=fake_zip(missing_day="2013-10-17")
        with patch.object(m,"ORIGINAL_SHA",m.pub.sha(data)):
            with self.assertRaises(m.EvidenceError):m.records_from_original(data)
    def test_bad_model_cell_is_rejected(self):
        model=m.fit({"ZERO|ZERO":[3,25]})
        with self.assertRaises(m.EvidenceError):m.risk(model,"FUTURE_LABEL")
    def test_training_requires_both_classes(self):
        with self.assertRaises(m.EvidenceError):m.fit({"ZERO|ZERO":[0,44]})

if __name__=="__main__":
    unittest.main()
