"""Constructed day/identity history tests, no actual Backblaze 2014 records."""
from datetime import date,timedelta
import importlib.util
from pathlib import Path
import sys
import unittest
ROOT=Path(__file__).resolve().parents[1]/"specialist_models"
spec=importlib.util.spec_from_file_location("model02_backblaze_2014_protocol",ROOT/"model02_backblaze_2014_protocol.py")
m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
def history(day,days=20,failure_offset=None,missing=None):
    h={}
    for k in range(days):
        when=day+timedelta(days=k)
        if when==missing:continue
        h[when]=(failure_offset==k)
        if failure_offset==k:break
    return h
class FrozenNextCohortRules(unittest.TestCase):
    def test_single_eligibility_history_does_not_peek_future(self):
        start=date(2014,2,1);s=history(start)
        self.assertFalse(m.admitted(s,start+timedelta(days=5)))
        self.assertTrue(m.admitted(s,start+timedelta(days=6)))
        later=dict(s);later[start+timedelta(days=10)]=True
        self.assertTrue(m.admitted(later,start+timedelta(days=6)))
    def test_true_delayed_failure_flag(self):
        start=date(2014,2,1)
        s=history(start,days=19,failure_offset=10)
        self.assertEqual(m.fate(s,start+timedelta(days=6)),"RECORDED_FAILURE")
    def test_missing_follow_up_is_censored(self):
        start=date(2014,2,1)
        s=history(start,days=17,missing=start+timedelta(days=10))
        self.assertEqual(m.fate(s,start+timedelta(days=6)),"RIGHT_CENSORED")
    def test_complete_nonfailure_7_days(self):
        start=date(2014,2,1)
        s=history(start)
        self.assertEqual(m.fate(s,start+timedelta(days=6)),"OBSERVED_NO_FAILURE")
    def test_fit_boundary_seals_future_even_if_present(self):
        end=date(2014,6,30)
        s={end-timedelta(days=i):False for i in range(7)}
        s[date(2014,7,1)]=True
        self.assertEqual(m.fate(s,end),"RIGHT_CENSORED")
    def test_cal_boundary_seals_future(self):
        d=date(2014,8,31)
        s={d-timedelta(days=i):False for i in range(7)}
        s[date(2014,9,1)]=True
        self.assertEqual(m.fate(s,d),"RIGHT_CENSORED")
    def test_dev_boundary_seals_test(self):
        d=date(2014,10,31)
        s={d-timedelta(days=i):False for i in range(7)}
        s[date(2014,11,1)]=True
        self.assertEqual(m.fate(s,d),"RIGHT_CENSORED")
    def test_protected_test_scoring_denied(self):
        d=date(2014,11,10)
        s={d-timedelta(days=i):False for i in range(7)}
        with self.assertRaises(m.ProtocolError):m.fate(s,d)
    def test_registration_definitively_blocks_promotion(self):
        p=m.frozen_registration()
        self.assertFalse(p["original_publisher_source_acquired"])
        self.assertIsNone(p["publisher_archive_sha256"])
        self.assertFalse(p["heldout_TEST_access_authorized"])
        self.assertFalse(p["live_inference_and_drive_replacement_authorized"])
        self.assertFalse(m.training_allowed(p))
        p["original_publisher_source_acquired"]=True
        self.assertFalse(m.training_allowed(p))
    def test_partitions_nonoverlap(self):
        self.assertEqual(m.phase(date(2014,3,1)),"FIT")
        self.assertEqual(m.phase(date(2014,8,1)),"CAL")
        self.assertEqual(m.phase(date(2014,10,1)),"EXPOSED_DEV")
        self.assertEqual(m.phase(date(2014,12,1)),"PROTECTED_TEST")
    def test_illegal_history_nonboolean_denied(self):
        d=date(2014,3,1);h={d-timedelta(days=i):0 for i in range(7)}
        self.assertFalse(m.admitted(h,d))
        with self.assertRaises(m.ProtocolError):m.fate(h,d)
if __name__=="__main__":unittest.main()
