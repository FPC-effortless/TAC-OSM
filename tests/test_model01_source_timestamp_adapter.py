"""Constructed publisher-grid source adapter invariants, no new performance claims."""
from datetime import datetime,timedelta
import importlib.util
from pathlib import Path
import sys
import unittest

ROOT=Path(__file__).resolve().parents[1]/"specialist_models"
sys.path.insert(0,str(ROOT))
spec=importlib.util.spec_from_file_location("model01_source_timestamp_adapter",
                                            ROOT/"model01_source_timestamp_adapter.py")
a=importlib.util.module_from_spec(spec)
sys.modules[spec.name]=a;spec.loader.exec_module(a)


def sample(end=200):
    start=a.SOURCE_ORIGIN+a.STEP*(end-144)
    return {"schema":a.SOURCE_SCHEMA,"source_series_id":a.SOURCE_ID,
            "observations":[{"observed_at":(start+a.STEP*i).strftime(a.FMT),
                             "energy_wh":float(60+i%9)} for i in range(145)]}


class OriginalSourceDateAdapterTests(unittest.TestCase):
    def test_original_date_maps_to_exact_source_phase(self):
        request=a.adapt(sample(end=200))
        self.assertEqual(request["source_aligned_step_index"],200)
        self.assertEqual(request["last_145_energy_wh"][-1],60+144%9)

    def test_exact_source_start_at_145th_observation(self):
        result=a.adapt(sample(end=144))
        self.assertEqual(result["source_aligned_step_index"],144)

    def test_original_checkpoint_receives_adapter_input(self):
        model,q=a.load_original()
        out=a.predict(model,q,a.adapt(sample(end=1500)))
        self.assertTrue(out["prediction_wh"]>=0)
        self.assertEqual(out["independent_qualification"],"NOT_QUALIFIED")

    def test_missing_time_or_reordered_observation_blocked(self):
        inp=sample()
        inp["observations"][42]["observed_at"]=inp["observations"][41]["observed_at"]
        with self.assertRaises(a.InputError):a.adapt(inp)

    def test_dropped_observation_blocked(self):
        inp=sample()
        inp["observations"].pop(40)
        with self.assertRaises(a.InputError):a.adapt(inp)

    def test_offset_timezone_and_subminute_blocked(self):
        for bad in ("2016-01-11T17:00:00Z","2016-01-11 17:00:01",
                    "2016-01-11 17:00:00+00:00","not a date"):
            with self.subTest(bad=bad),self.assertRaises(a.InputError):
                a.parse_source_time(bad)

    def test_no_other_publisher_or_hidden_outcome(self):
        inp=sample()
        inp["source_series_id"]="SOME_OTHER_BUILDING"
        with self.assertRaises(a.InputError):a.adapt(inp)
        inp=sample()
        inp["observations"][0]["future_energy"]=500
        with self.assertRaises(a.InputError):a.adapt(inp)

    def test_nonfinite_negative_wrong_types_denied(self):
        for value in (float("nan"),float("inf"),-1,"80",True):
            with self.subTest(value=str(value)):
                inp=sample()
                inp["observations"][22]["energy_wh"]=value
                with self.assertRaises(a.InputError):a.adapt(inp)

    def test_original_origin_before_145_samples_denied(self):
        inp=sample(end=144)
        shift=timedelta(minutes=10)
        for row in inp["observations"]:
            t=a.parse_source_time(row["observed_at"])-shift
            row["observed_at"]=t.strftime(a.FMT)
        with self.assertRaises(a.InputError):a.adapt(inp)

    def test_out_of_source_bounds_denied(self):
        inp=sample(end=20_000_001)
        with self.assertRaises(a.InputError):a.adapt(inp)

    def test_provided_step_index_cannot_override_timestamp_source(self):
        inp=sample()
        inp["source_aligned_step_index"]=0
        with self.assertRaises(a.InputError):a.adapt(inp)


if __name__=="__main__":
    unittest.main()
