"""Constructed data only: source-fit graph, phase/censoring and JSON replay."""
from datetime import timedelta
import csv
import importlib.util
import io
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
import zipfile

BASE=Path(__file__).resolve().parents[1]/"specialist_models"
sys.path.insert(0,str(BASE))
spec=importlib.util.spec_from_file_location("bb_classical",BASE/"backblaze_2013_classical_dev_fit.py")
m=importlib.util.module_from_spec(spec)
sys.modules[spec.name]=m
spec.loader.exec_module(m)

def archive(*,bad_attr=False,missing_date=False):
    raw=io.BytesIO()
    with zipfile.ZipFile(raw,"w",zipfile.ZIP_DEFLATED) as z:
        for i,d in enumerate(m.source.days()):
            if missing_date and i==12:
                continue
            f=io.StringIO();writer=csv.writer(f)
            writer.writerow(["date","serial_number","model","failure",
                             "smart_5_raw","smart_9_raw","smart_197_raw"])
            for j in range(20):
                failure_day=20 if j<3 else (41 if j<5 else None)
                if failure_day is not None and i>failure_day:
                    continue
                failure=failure_day == i
                signal = 100 if failure_day is not None and 0 <= failure_day-i <= 4 else 0
                attr = "UNTRUSTED" if bad_attr and i==10 and j==2 else str(signal)
                writer.writerow([d.isoformat(),f"SERIAL-{j:02d}","MODEL-A",
                                 int(failure),attr,str(100+i+j),str(signal)])
            z.writestr("source/"+d.isoformat()+".csv",f.getvalue())
        # Intentionally malformed future member must remain unopened.
        z.writestr("source/2013-11-16.csv","PROTECTED_FUTURE_SENTINEL")
    return raw.getvalue()

def patched_hash(data):
    expected=m.original.sha(data)
    return (patch.object(m,"SOURCE_SHA",expected),
            patch.object(m.source,"SOURCE_SHA",expected),
            patch.object(m.admission.source,"SOURCE_SHA",expected))

class ClassicalBackblazeFixtureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data=archive()
        a,b,c=patched_hash(cls.data)
        with a,b,c:
            cls.cohorts,cls.days=m.cohorts(cls.data)

    def test_fit_and_dev_have_independent_calendar_phase_labels(self):
        fit,dev=self.cohorts["FIT"],self.cohorts["DEV"]
        self.assertEqual(fit["unique_drives_with_positive_windows"],3)
        self.assertEqual(dev["unique_drives_with_positive_windows"],2)
        self.assertTrue(sum(fit["ys"])>0)
        self.assertTrue(sum(dev["ys"])>0)
        self.assertEqual(len(self.days),46)

    def test_nonindependent_overlapping_positive_windows_declared(self):
        fit,dev=self.cohorts["FIT"],self.cohorts["DEV"]
        self.assertGreater(sum(fit["ys"]),fit["unique_drives_with_positive_windows"])
        self.assertGreater(sum(dev["ys"]),dev["unique_drives_with_positive_windows"])

    def test_fit_replay_and_calibration_not_established(self):
        model=m.fit(self.cohorts["FIT"])
        original=json.loads(json.dumps(model))
        out=m.evaluate(model,self.cohorts,self.days)
        self.assertEqual(out["conclusion"],"DEV_ONLY_NO_REAL_WORLD_QUALIFICATION")
        self.assertFalse(out["independent_evaluation"])
        self.assertEqual(out["dev"]["positive_windows"],sum(self.cohorts["DEV"]["ys"]))
        self.assertEqual(out["fit_unique_positive_physical_drives"],3)
        self.assertEqual(out["dev_unique_positive_physical_drives"],2)
        self.assertEqual([m.predict(model,x) for x in self.cohorts["DEV"]["xs"][:20]],
                         [m.predict(original,x) for x in self.cohorts["DEV"]["xs"][:20]])

    def test_predict_never_uses_hidden_target(self):
        model=m.fit(self.cohorts["FIT"])
        x=self.cohorts["DEV"]["xs"][0]
        pred=m.predict(model,x)
        self.assertTrue(0<pred<1)
        with self.assertRaises(m.StudyError):
            m.predict(model,x+[float("nan")])

    def test_missing_or_corrupted_source_blocks(self):
        d=archive(missing_date=True)
        a,b,c=patched_hash(d)
        with a,b,c,self.assertRaises(m.admission.SourceError):
            m.cohorts(d)
        d=archive(bad_attr=True)
        a,b,c=patched_hash(d)
        with a,b,c,self.assertRaises(m.StudyError):
            m.cohorts(d)

    def test_original_source_sha_enforced(self):
        with self.assertRaises(m.StudyError):
            m.cohorts(self.data)

    def test_no_genuine_score_claim(self):
        model=m.fit(self.cohorts["FIT"])
        result=m.evaluate(model,self.cohorts,self.days)
        self.assertFalse(result["source_published_asof_verified"])
        self.assertFalse(result["trained_on_protected_TEST"])
        self.assertIn("CLASSICAL_NOT_TAC",result["model_type"])
        self.assertGreater(result["dev_censored_windows_excluded"],0)

if __name__ == "__main__":
    unittest.main()
