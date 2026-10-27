"""Constructed 46-day publisher archive fixtures; never external source success."""
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
spec=importlib.util.spec_from_file_location("backblaze_2013_cohort_feasibility",
                                            ROOT/"backblaze_2013_cohort_feasibility.py")
m=importlib.util.module_from_spec(spec)
sys.modules[spec.name]=m
spec.loader.exec_module(m)

def archive(*,resurrect=False,corrupt_inside=False,skip_day=None,path_traversal=False):
    buff=io.BytesIO()
    with zipfile.ZipFile(buff,"w",zipfile.ZIP_DEFLATED) as z:
        for i,day in enumerate(m.days()):
            label=day.isoformat()
            if label==skip_day:continue
            text=io.StringIO()
            w=csv.writer(text)
            w.writerow(["date","serial_number","model","failure","smart_5_raw"])
            if i<=8 or (resurrect and i==9):
                w.writerow([label,"A","MODEL-A","1" if i==8 else "0","2"])
            w.writerow([label,"B","MODEL-B","0","5"])
            if i<=19:
                w.writerow([label,"C","MODEL-C","0",""])
            if i>15:
                w.writerow([label,"LATER_ONLY","MODEL-X","0","1"])
            z.writestr("2013/"+label+".csv",text.getvalue())
        z.writestr("2013/2013-11-16.csv","INTENTIONALLY_INVALID_HIDDEN_TEST_CONTENT")
        if path_traversal:z.writestr("../bad",b"bad")
    return buff.getvalue()

def with_pinned(data):
    return patch.object(m,"SOURCE_SHA",m.original.sha(data))


class RetrospectiveCohortFixtureTests(unittest.TestCase):
    def test_window_and_outcome_census(self):
        data=archive()
        with with_pinned(data):
            res=m.count_source(data)
        self.assertEqual(res["initial_functioning_enrolled_drives"],3)
        self.assertEqual(res["publisher_original_2013_csv_days_parsed"],46)
        self.assertEqual(res["recorded_failure_flags_all_46_source_days"],1)
        self.assertEqual(res["enrolled_unique_failure_drives_46_days"],1)
        fit=res["FIT_source_feasibility_exposed"]
        self.assertEqual(fit["provisional_positive_windows"],7)
        self.assertGreater(fit["right_censored_windows"],0)
        self.assertGreater(fit["complete_negative_windows"],0)
        self.assertEqual(fit["source_observation_windows"],
                         fit["provisional_positive_windows"]+fit["complete_negative_windows"]+fit["right_censored_windows"])
        self.assertEqual(res["DEV_source_feasibility_exposed"]["provisional_positive_windows"],0)
        self.assertFalse(res["heldout_evaluation_performed"])
    def test_missing_publisher_day_blocks(self):
        data=archive(skip_day="2013-10-03")
        with with_pinned(data),self.assertRaises(m.CohortError):m.count_source(data)
    def test_untrusted_zip_path_blocks(self):
        data=archive(path_traversal=True)
        with with_pinned(data),self.assertRaises(m.CohortError):m.count_source(data)
    def test_resurrection_after_original_failure_blocks(self):
        data=archive(resurrect=True)
        with with_pinned(data),self.assertRaises(m.CohortError):m.count_source(data)
    def test_original_SHA_must_match(self):
        data=archive()
        with self.assertRaises(m.CohortError):m.count_source(data)
    def test_strict_disclosed_source_period(self):
        self.assertEqual(m.days()[0].isoformat(),"2013-10-01")
        self.assertEqual(m.days()[-1].isoformat(),"2013-11-15")
        self.assertNotIn("2013-11-16",[d.isoformat() for d in m.days()])
    def test_censoring_dev_boundary_even_if_zip_contains_later_data(self):
        data=archive()
        with with_pinned(data):
            result=m.count_source(data)
        dev=result["DEV_source_feasibility_exposed"]
        self.assertGreater(dev["right_censored_windows"],0)
        self.assertFalse(result["trained_model"])
        self.assertFalse(result["actual_source_time_availability_verified"])
    def test_missing_smart_feature_is_counted_not_imputed_as_zero(self):
        data=archive()
        with with_pinned(data):
            result=m.count_source(data)
        self.assertGreater(result["missing_selected_smart_values_total"]["smart_198_raw"],0)
        self.assertGreater(result["missing_selected_smart_values_total"]["smart_5_raw"],0)

if __name__=="__main__":
    unittest.main()
