"""Constructed source ZIP fixtures; NOT Backblaze genuine data or learned skill."""
import csv
from datetime import timedelta
import importlib.util
import io
from pathlib import Path
import sys
import unittest
import zipfile

PATH=Path(__file__).resolve().parents[1]/"specialist_models"/"backblaze_2013_source_probe.py"
spec=importlib.util.spec_from_file_location("bb_probe",PATH)
m=importlib.util.module_from_spec(spec)
sys.modules[spec.name]=m
spec.loader.exec_module(m)


def fake_zip(missing=None,duplicate=None,wrong_date=False,evil=False):
    buf=io.BytesIO()
    with zipfile.ZipFile(buf,"w",zipfile.ZIP_DEFLATED) as z:
        for i in range(m.DAYS):
            d=(m.FIRST+timedelta(days=i)).isoformat()
            name=d+".csv"
            if d==missing:continue
            contents=io.StringIO()
            writer=csv.writer(contents)
            writer.writerow(["date","serial_number","model","failure","smart_5_raw","smart_9_raw"])
            for j in range(3):
                writer.writerow([d if not wrong_date or i!=0 else "2013-09-29",
                    "SERIAL-"+str(j),"MODEL-X","1" if i==2 and j==1 else "0",
                    "1" if j!=0 else "",str(50+i)])
            z.writestr("daily/"+name,contents.getvalue())
            if d==duplicate:z.writestr("duplicate/"+name,contents.getvalue())
        if evil:z.writestr("../unsafe.txt","evil")
    return buf.getvalue()


class Original2013ProbeFixtureTests(unittest.TestCase):
    def test_bounded_original_source_receipt(self):
        x=m.scan(fake_zip())
        self.assertEqual(x["total_daily_rows"],3*m.DAYS)
        self.assertEqual(x["total_recorded_failure_flags"],1)
        self.assertFalse(x["fit_labels_created"])
        self.assertFalse(x["source_publication_time_verified"])
        self.assertFalse(x["trained_tac_model"])
        self.assertEqual(x["daily"][0]["missing_fields_by_column"]["smart_197_raw"],3)
    def test_missing_publisher_day_blocks(self):
        with self.assertRaises(m.SourceError):
            m.scan(fake_zip(missing=m.FIRST.isoformat()))
    def test_duplicate_member_name_blocks(self):
        with self.assertRaises(m.SourceError):
            m.scan(fake_zip(duplicate=m.FIRST.isoformat()))
    def test_wrong_original_date_blocks(self):
        with self.assertRaises(m.SourceError):
            m.scan(fake_zip(wrong_date=True))
    def test_untrusted_zip_path_blocks(self):
        with self.assertRaises(m.SourceError):
            m.scan(fake_zip(evil=True))
    def test_empty_or_invalid_zip_blocks(self):
        with self.assertRaises(m.SourceError):m.scan(b"")
        with self.assertRaises(zipfile.BadZipFile):m.scan(b"not a zip")
    def test_original_source_digest_not_hash_of_selected_rows(self):
        src=fake_zip()
        x=m.scan(src)
        self.assertEqual(x["original_zip_sha256"],m.sha(src))
        self.assertEqual(len(x["daily"][0]["original_member_sha256"]),64)

if __name__=="__main__":
    unittest.main()
