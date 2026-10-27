"""Constructed test records only. Not a real sensor/model evaluation."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

PATH = Path(__file__).resolve().parents[1] / "specialist_models" / "forecast_reliability.py"
spec = importlib.util.spec_from_file_location("forecast_reliability", PATH)
m = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = m
spec.loader.exec_module(m)

def contract():
    return {"schema": "forecast-interval-contract-v1", "model_sha256": "a" * 64,
            "source_id": "CONSTRUCTED_TEST_ONLY", "model_fit_end": "2024-12-31T00:00:00Z",
            "calibration_start": "2025-01-01T00:00:00Z",
            "calibration_end": "2025-03-01T00:00:00Z",
            "evaluation_start": "2025-03-02T00:00:00Z",
            "alpha": 0.1, "min_cell_n": 20,
            "channels": ["pm25", "no2"], "gap_kinds": ["gap", "ordinary"],
            "horizons_hours": [6]}

def record(i, channel="pm25", gap="gap", issued="2025-01-05T00:00:00Z", y=6.):
    return {"row_id":f"row-{channel}-{gap}-{i}", "entity_id":f"site-{channel}-{gap}-{i}",
            "issued_at":issued,"model_sha256":"a"*64,"channel":channel,
            "gap_kind":gap,"horizon_hours":6,"point":5.,"lower":4.,"upper":6.,
            "truth":y,"outcome_available_at":"2025-01-06T00:00:00Z" if issued.startswith("2025-01")
                                        else "2025-03-04T00:00:00Z"}

def calibration_rows():
    return [record(i,c,g) for c in ("pm25","no2") for g in ("gap","ordinary") for i in range(25)]

class ForecastReliabilityFixtureTests(unittest.TestCase):
    def test_calibration_every_required_group(self):
        a=m.calibrate(contract(),calibration_rows(),"b"*64)
        self.assertEqual(a["evidence_status"],"CALIBRATION_ONLY_UNVERIFIED_SOURCE")
        for v in a["cell_corrections"].values():
            self.assertEqual(v["n"],25)
            self.assertEqual(v["correction"],0.)
    def test_widen_for_outlying_source_labels(self):
        rows=calibration_rows()
        for x in rows:
            x["truth"]=10.
        a=m.calibrate(contract(),rows,"b"*64)
        self.assertEqual(a["cell_corrections"][m.cell_key("pm25",6,"gap")]["correction"],4.)
        req={k:v for k,v in record(101, issued="2025-03-03T00:00:00Z").items()
             if k not in m.LABEL}
        out=m.adjusted(a,req)
        self.assertEqual((out["lower"],out["upper"]),(0.,10.))
        self.assertEqual(out["qualification"],"UNVERIFIED_TEMPORAL_CALIBRATION")
    def test_source_time_future_label_is_blocked(self):
        rows=calibration_rows()
        rows[0]["outcome_available_at"]="2025-03-02T00:00:00Z"
        with self.assertRaises(m.EvidenceError):m.calibrate(contract(),rows,"b"*64)
    def test_explicit_fit_cal_test_order_required(self):
        x=contract();x["calibration_start"]=x["model_fit_end"]
        with self.assertRaises(m.EvidenceError):m.valid_config(x)
    def test_missing_critical_group_blocks(self):
        rows=[r for r in calibration_rows() if r["gap_kind"]!="gap"]
        with self.assertRaises(m.EvidenceError):m.calibrate(contract(),rows,"b"*64)
    def test_duplicate_entity_horizon_blocks(self):
        rows=calibration_rows()
        rows[1]["entity_id"]=rows[0]["entity_id"]
        with self.assertRaises(m.EvidenceError):m.calibrate(contract(),rows,"b"*64)
    def test_cannot_calibrate_on_evaluation_period(self):
        rows=calibration_rows()
        rows[0]["issued_at"]="2025-03-05T00:00:00Z"
        with self.assertRaises(m.EvidenceError):m.calibrate(contract(),rows,"b"*64)
    def test_cannot_infer_from_hidden_target(self):
        artifact=m.calibrate(contract(),calibration_rows(),"b"*64)
        with self.assertRaises(m.EvidenceError):
            m.adjusted(artifact,record(105,issued="2025-03-03T00:00:00Z"))
    def test_wrong_checkpoint_digest_rejected(self):
        a=m.calibrate(contract(),calibration_rows(),"b"*64)
        req={k:v for k,v in record(111,issued="2025-03-03T00:00:00Z").items() if k not in m.LABEL}
        req["model_sha256"]="f"*64
        with self.assertRaises(m.EvidenceError):m.adjusted(a,req)
    def test_evaluation_not_qualification(self):
        a=m.calibrate(contract(),calibration_rows(),"b"*64)
        rows=[record(i,c,g,issued="2025-03-03T00:00:00Z") for c in ("pm25","no2")
              for g in ("gap","ordinary") for i in range(4)]
        out=m.evaluate(a,rows)
        self.assertEqual(out["status"],"DESCRIPTIVE_EVALUATION_ONLY")
        self.assertFalse(out["independent_cohort_certified"])
        self.assertEqual(out["cells"][m.cell_key("pm25",6,"gap")]["n"],4)
    def test_interval_score_penalizes_misses(self):
        self.assertGreater(m.score_interval(4.,6.,10.,.1),m.score_interval(4.,6.,5.,.1))
    def test_finite_rank_requires_calibration_count(self):
        a=contract();a["alpha"]=.001
        with self.assertRaises(m.EvidenceError):m.calibrate(a,calibration_rows(),"b"*64)
    def test_unknown_field_and_unsafe_replay_denied(self):
        rows=calibration_rows()
        rows[0]["future_target_cache"]=42
        with self.assertRaises(m.EvidenceError):m.calibrate(contract(),rows,"b"*64)

if __name__=="__main__":
    unittest.main()
