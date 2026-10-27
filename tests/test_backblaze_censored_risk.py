"""Constructed daily CSV diagnostics, no actual Backblaze download or model fit."""
import csv
from datetime import date, timedelta
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

PATH = Path(__file__).resolve().parents[1] / "specialist_models" / "backblaze_censored_risk.py"
spec = importlib.util.spec_from_file_location("backblaze_censored_risk", PATH)
m = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = m
spec.loader.exec_module(m)


class BackblazeFixtureTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.start = date(2026, 1, 1)
        self.files = []
        for i in range(12):
            d = self.start + timedelta(days=i)
            entries = []
            if i < 5:
                entries.append(["A", "MODEL-A", "1" if i == 4 else "0", str(i), str(100 + i)])
            entries.append(["B", "MODEL-B", "0", str(i + 5), ""])
            if i < 3:
                entries.append(["C", "MODEL-C", "0", "", ""])
            path = self.root / (d.isoformat() + ".csv")
            with path.open("w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["date", "serial_number", "model", "failure", "smart_5_raw", "smart_9_raw"])
                for r in entries:
                    w.writerow([d.isoformat()] + r)
            self.files.append({"date":d.isoformat(), "path":path.name, "sha256":m.sha256(path)})
        self.manifest = {
            "schema":"backblaze-entity-cohort-v1",
            "source_url":m.OFFICIAL_SOURCE,
            "permission":"review-required-no-redistribution",
            "enrollment_date":"2026-01-01",
            "fit_end":"2026-01-08", "dev_end":"2026-01-13",
            "horizon_days":3,
            "smart_fields":["smart_5_raw","smart_9_raw"],
            "enrolled_drives":[{"serial_number":x,"model":"MODEL-"+x} for x in ("A","B","C")],
            "csv_files":self.files,
        }

    def load(self, phase="fit"):
        return m.load_selected(self.manifest,phase,self.root)

    def test_realistic_censoring_missing_drive_not_negative(self):
        history,_=self.load()
        examples,idx,censored=m.make_rows(self.manifest,history,"fit")
        positives=[x for x in idx if x["label_status"]=="POSITIVE"]
        negatives=[x for x in idx if x["label_status"]=="NEGATIVE"]
        self.assertTrue(positives)
        self.assertTrue(negatives)
        self.assertTrue(any(x["label_status"]=="RIGHT_CENSORED" for x in censored))
        self.assertTrue(any(x["model"]=="MODEL-C" for x in censored))
        self.assertFalse(any(x["model"]=="MODEL-C" for x in idx))
        self.assertTrue(all(e["outcome"] in (0,1) for e in examples))
    def test_censored_source_rows_train_and_score_through_typed_risk_baseline(self):
        """Constructed E2E integration: no actual Backblaze record or capability evidence."""
        typed_path=Path(__file__).resolve().parents[1]/"specialist_models"/"typed_event_risk.py"
        typed_spec=importlib.util.spec_from_file_location("typed_event_risk",typed_path)
        typed=importlib.util.module_from_spec(typed_spec)
        sys.modules[typed_spec.name]=typed
        typed_spec.loader.exec_module(typed)
        history,_=self.load("dev")
        fit_rows,fit_index,fit_censored=m.make_rows(self.manifest,history,"fit")
        dev_rows,dev_index,dev_censored=m.make_rows(self.manifest,history,"dev")
        self.assertTrue(any(x["outcome"]==1 for x in fit_rows))
        self.assertTrue(any(x["outcome"]==0 for x in fit_rows))
        source={"features":sorted(fit_rows[0]["features"]),
                "status":"source-claimed-not-independently-audited",
                "fit_cutoff":"2026-01-08T00:00:00Z",
                "source_url":"https://example.org/CONSTRUCTED_NOT_REAL",
                "dataset_license":"constructed test only",
                "data_steward":"constructed test only"}
        model=typed.fit(fit_rows,source,"a"*64)
        self.assertEqual(model["qualification"],"DEVELOPMENT_ONLY_NOT_CALIBRATED")
        report=typed.evaluate(model,dev_rows)
        self.assertEqual(report["qualification"],"DEV_ONLY_NOT_INDEPENDENT_TEST")
        self.assertEqual(report["cases"],len(dev_index))
        self.assertTrue(0 <= report["brier"] <= 1)
        self.assertTrue(dev_censored)
        self.assertTrue(fit_censored)
    def test_first_positive_label_is_after_prediction(self):
        timelines,_=self.load()
        examples,idx,_=m.make_rows(self.manifest,timelines,"fit")
        pos=[examples[k] for k,v in enumerate(idx) if v["label_status"]=="POSITIVE"]
        self.assertTrue(pos)
        self.assertTrue(all(m.end_of_day(self.start) < x["outcome_at"] for x in pos))
    def test_asof_feature_values_and_missing_flags(self):
        timelines,_=self.load()
        xs,idx,_=m.make_rows(self.manifest,timelines,"fit")
        entry=next(xs[k] for k,d in enumerate(idx) if d["model"]=="MODEL-B"
                   and d["observation_day"]=="2026-01-03")
        self.assertEqual(entry["features"]["smart_9_raw_present"]["value"],0)
        self.assertEqual(entry["features"]["smart_9_raw_log1p"]["value"],0.)
        self.assertEqual(entry["features"]["smart_5_raw_delta_present"]["value"],1)
        self.assertEqual(entry["features"]["smart_5_raw_log1p"]["available_at"],entry["observed_at"])
        self.assertEqual(set(entry),{"observed_at","features","outcome","outcome_at"})
    def test_dev_uses_only_pretest_outcomes(self):
        timelines,_=self.load("dev")
        xs,idx,cens=m.make_rows(self.manifest,timelines,"dev")
        self.assertTrue(any(x["model"]=="MODEL-B" for x in cens))
        self.assertTrue(xs)
        self.assertTrue(all(y["outcome_day"] < self.manifest["dev_end"] for y in idx))
    def test_future_dev_data_not_loaded_in_fit(self):
        timelines,_=self.load()
        self.assertEqual(max(timelines[("B","MODEL-B")]),date(2026,1,7))
    def test_tampered_publisher_bytes_blocked(self):
        p=self.root/"2026-01-02.csv"
        p.write_text(p.read_text().replace("MODEL-B","INJECTED"),encoding="utf-8")
        with self.assertRaises(m.SourceError):self.load()
    def test_missing_daily_source_file_blocks(self):
        self.manifest["csv_files"]=self.files[:-1]
        with self.assertRaises(m.SourceError):self.load("dev")
    def test_duplicate_enrolled_identites_block(self):
        self.manifest["enrolled_drives"].append(self.manifest["enrolled_drives"][0])
        with self.assertRaises(m.SourceError):self.load()
    def test_future_protected_csv_block(self):
        future=self.files[-1].copy()
        future["date"]="2026-01-13"
        future["path"]="2026-01-13.csv"
        self.manifest["csv_files"].append(future)
        with self.assertRaises(m.SourceError):self.load()
    def test_resurrected_failed_drive_blocks(self):
        p=self.root/"2026-01-07.csv"
        lines=p.read_text().splitlines()
        lines.append("2026-01-07,A,MODEL-A,0,10,10")
        p.write_text("\n".join(lines)+"\n")
        self.files[6]["sha256"]=m.sha256(p)
        with self.assertRaises(m.SourceError):self.load()
    def test_reject_alternative_nonpublisher_source(self):
        self.manifest["source_url"]="https://mirror.example.org"
        with self.assertRaises(m.SourceError):self.load()
    def test_input_file_path_traversal_rejected(self):
        self.files[0]["path"]="../fake.csv"
        with self.assertRaises(m.SourceError):self.load()
    def test_no_fit_outcome_can_depend_on_dev(self):
        timelines,_=self.load()
        xs,idx,censor=m.make_rows(self.manifest,timelines,"fit")
        self.assertTrue(all(d["outcome_day"] < self.manifest["fit_end"] for d in idx))
    def test_missing_smart_does_not_mean_failure(self):
        timeline={date(2026,1,i):{"failed":False,"smart":{"smart_5_raw":None}}
                  for i in range(1,4)}
        status,y,_=m.future_status(timeline,date(2026,1,1),2)
        self.assertEqual((status,y),("NEGATIVE",0))
        status,y,_=m.future_status(timeline,date(2026,1,2),2)
        self.assertEqual((status,y),("RIGHT_CENSORED",None))

if __name__=="__main__":
    unittest.main()
