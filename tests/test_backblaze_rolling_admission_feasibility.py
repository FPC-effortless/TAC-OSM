"""Constructed rolling entity admission and censor tests; no real skill metrics."""
from datetime import timedelta
from io import BytesIO,StringIO
import csv
import importlib.util
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
import zipfile

ROOT=Path(__file__).resolve().parents[1]/"specialist_models"
sys.path.insert(0,str(ROOT))
spec=importlib.util.spec_from_file_location("bb_rolling",ROOT/"backblaze_rolling_admission_feasibility.py")
m=importlib.util.module_from_spec(spec)
sys.modules[spec.name]=m
spec.loader.exec_module(m)

def constructed_zip(*,bad_future=False,missing_day=None,resurrect=False,path_attack=False):
    buf=BytesIO()
    with zipfile.ZipFile(buf,"w",zipfile.ZIP_DEFLATED) as z:
        for idx,d in enumerate(m.source.days()):
            date=d.isoformat()
            if date==missing_day:continue
            io_=StringIO();w=csv.writer(io_)
            w.writerow(["date","serial_number","model","failure","smart_5_raw"])
            if idx<=8 or (resurrect and idx==9):
                w.writerow([date,"A","A-MODEL",int(idx==8),0])
            w.writerow([date,"B","B-MODEL",0,1])
            if idx<=19:w.writerow([date,"C","C-MODEL",0,2])
            if 30<=idx<=40:
                w.writerow([date,"NEW","NEW-MODEL",int(idx==40),3])
            z.writestr("prefix/"+date+".csv",io_.getvalue())
        z.writestr("prefix/2013-11-16.csv","malformed_hidden" if bad_future else "date,serial_number,model,failure\n")
        if path_attack:z.writestr("../bad.txt","bad")
    return buf.getvalue()

def patch_sha(payload):
    return patch.object(m.source,"SOURCE_SHA",m.original.sha(payload))

class DynamicEnrolmentFixtureTests(unittest.TestCase):
    def test_rolling_7_day_past_only_admission_and_exposed_dev_failure(self):
        data=constructed_zip(bad_future=True)
        with patch_sha(data):
            timelines,daily=m.read_timeline(data)
            report=m.audit(timelines,daily)
        self.assertEqual(report["publisher_failure_flags"],2)
        self.assertEqual(report["distinct_publisher_entity_ids"],4)
        self.assertEqual(report["DEV_exposed"]["unique_drives_with_future_failure_windows"],1)
        self.assertGreater(report["DEV_exposed"]["observed_future_failure_windows"],0)
        self.assertFalse(report["protected_future_content_parsed"])
        self.assertFalse(report["trained_model"])
    def test_seven_day_warmup_uses_only_past_and_current(self):
        data=constructed_zip()
        with patch_sha(data):
            timelines,_=m.read_timeline(data)
        key=("NEW","NEW-MODEL")
        self.assertFalse(m.admitted_at(timelines[key],m.source.FIRST+timedelta(days=35)))
        self.assertTrue(m.admitted_at(timelines[key],m.source.FIRST+timedelta(days=36)))
    def test_fit_phase_does_not_promote_november_failure(self):
        data=constructed_zip()
        with patch_sha(data):
            timelines,daily=m.read_timeline(data)
            result=m.audit(timelines,daily)
        self.assertEqual(result["FIT_exposed"]["unique_drives_with_future_failure_windows"],1)
        self.assertEqual(result["DEV_exposed"]["unique_drives_with_future_failure_windows"],1)
    def test_resurrection_after_failure_blocks(self):
        data=constructed_zip(resurrect=True)
        with patch_sha(data),self.assertRaises(m.SourceError):
            m.read_timeline(data)
    def test_source_missing_date_blocks(self):
        data=constructed_zip(missing_day="2013-10-14")
        with patch_sha(data),self.assertRaises(m.SourceError):
            m.read_timeline(data)
    def test_zip_traversal_blocks(self):
        data=constructed_zip(path_attack=True)
        with patch_sha(data),self.assertRaises(m.SourceError):
            m.read_timeline(data)
    def test_sha_lock_blocks_untrusted_source(self):
        with self.assertRaises(m.SourceError):
            m.read_timeline(constructed_zip())

if __name__=="__main__":
    unittest.main()
