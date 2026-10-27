"""Constructed source clock and meter safety tests. NOT original model skill."""
from copy import deepcopy
from datetime import datetime,timezone,timedelta
import importlib.util
from pathlib import Path
import sys
import unittest

ROOT=Path(__file__).resolve().parents[1]/"specialist_models"
spec=importlib.util.spec_from_file_location("model01_meter_admission",ROOT/"model01_meter_admission.py")
m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)

T=datetime(2026,10,27,9,0,tzinfo=timezone.utc)
def stamp(t):
    return t.strftime("%Y-%m-%dT%H:%M:%SZ")

def req():
    last=T-timedelta(minutes=5)
    first=last-timedelta(minutes=1440)
    data=[]
    for i in range(145):
        d=first+timedelta(minutes=10*i)
        data.append({"window_end_utc":stamp(d),"available_at_utc":stamp(d+timedelta(minutes=1)),
                     "energy_wh":85.0+(i%3)})
    return {"schema":m.SCHEMA,"meter_id":"METER-X","target":m.TARGET,
            "sampling_minutes":10,"timezone":"UTC","decision_time_utc":stamp(T),
            "samples":data,"allow_advisory_persistence":True}

class MeterContractTests(unittest.TestCase):
    def test_valid_persistence_advisory_only(self):
        x=m.score_request(req())
        self.assertEqual(x["decision"],"ADVISORY_BASELINE_ONLY")
        self.assertEqual(x["prediction_wh"],req()["samples"][-1]["energy_wh"])
        self.assertFalse(x["original_ridge_invoked"])
        self.assertFalse(x["authorized_action"])
        self.assertIsNone(x["calibrated_interval"])
    def test_mismatched_whole_house_denied_as_invalid_model_target(self):
        r=req();r["target"]="whole_house_energy_wh_per_10min"
        result=m.score_request(r)
        self.assertEqual(result["decision"],"DEFER")
        self.assertEqual(result["reason"],"TARGET_NOT_APPLIANCES_WH_10MIN")
    def test_stale_sensor_deferred_without_model_call(self):
        r=req();r["decision_time_utc"]=stamp(T+timedelta(minutes=40))
        x=m.score_request(r)
        self.assertEqual(x["reason"],"LAST_COMPLETE_READING_STALE")
        self.assertFalse(x["model_invoked"])
    def test_missing_measurement_rejected(self):
        r=req();r["samples"][4]["energy_wh"]=None
        with self.assertRaises(m.AdmissionError):m.score_request(r)
    def test_missing_interval_fails(self):
        r=req();r["samples"][8]["window_end_utc"]=r["samples"][7]["window_end_utc"]
        with self.assertRaises(m.AdmissionError):m.score_request(r)
    def test_availability_future_leak_fails(self):
        r=req();r["samples"][40]["available_at_utc"]=stamp(T+timedelta(days=1))
        with self.assertRaises(m.AdmissionError):m.score_request(r)
    def test_available_before_complete_interval_fails(self):
        r=req();r["samples"][40]["available_at_utc"]=stamp(T-timedelta(days=5))
        with self.assertRaises(m.AdmissionError):m.score_request(r)
    def test_future_source_window_fails(self):
        r=req();r["samples"][-1]["window_end_utc"]=stamp(T+timedelta(minutes=10))
        with self.assertRaises(m.AdmissionError):m.score_request(r)
    def test_hidden_future_target_fails(self):
        r=req();r["future_true_energy_wh"]=300
        with self.assertRaises(m.AdmissionError):m.score_request(r)
    def test_not_explicitly_requesting_advisory_defers(self):
        r=req();r["allow_advisory_persistence"]=False
        x=m.score_request(r)
        self.assertEqual(x["decision"],"DEFER")
    def test_target_schema_is_exact(self):
        r=req();r["timezone"]="Africa/Lagos"
        with self.assertRaises(m.AdmissionError):m.score_request(r)
    def test_seconds_precision_and_timezone_strict(self):
        r=req();r["decision_time_utc"]="2026-10-27T09:00Z"
        with self.assertRaises(m.AdmissionError):m.score_request(r)
        r=req();r["decision_time_utc"]="2026-10-27T10:00:00+01:00"
        with self.assertRaises(m.AdmissionError):m.score_request(r)
    def test_numeric_boolean_and_nan_denied(self):
        for v in (True,float("nan"),float("inf"),-10):
            r=req();r["samples"][12]["energy_wh"]=v
            with self.assertRaises(m.AdmissionError):m.score_request(r)
    def test_invalid_meter_id_and_extra_fields_denied(self):
        r=req();r["meter_id"]=" ../../etc "
        with self.assertRaises(m.AdmissionError):m.score_request(r)
        r=req();r["samples"][0]["remote_model_decision"]="approved"
        with self.assertRaises(m.AdmissionError):m.score_request(r)
    def test_full_145_source_windows_required(self):
        r=req();r["samples"].pop()
        with self.assertRaises(m.AdmissionError):m.score_request(r)
if __name__=="__main__":unittest.main()
