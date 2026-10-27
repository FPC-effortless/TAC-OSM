"""Constructed UCI235 archive fixtures; no scientific performance from these tests."""
from datetime import timedelta
import importlib.util
import io
from pathlib import Path
import sys
import unittest
import zipfile

ROOT=Path(__file__).resolve().parents[1]/"specialist_models"
sys.path.insert(0,str(ROOT))
spec=importlib.util.spec_from_file_location("model01_uci235_transfer_stress",
                                            ROOT/"model01_uci235_transfer_stress.py")
s=importlib.util.module_from_spec(spec);sys.modules[spec.name]=s;spec.loader.exec_module(s)


def archive(*,missing_block=None,skip_minute=None,bad_header=False,future_garbage=False):
    out=io.StringIO()
    out.write("Other;Time;Global_active_power\n" if bad_header
              else "Date;Time;Global_active_power\n")
    d=s.START
    while d<s.END:
        if d!=skip_minute:
            key=d.strftime("%Y-%m-%d %H:%M:%S")
            datum="?" if missing_block==d.replace(minute=d.minute//10*10) else "0.96"
            out.write(d.strftime("%d/%m/%Y;%H:%M:%S;")+datum+"\n")
        d+=timedelta(minutes=1)
    if future_garbage:
        out.write(s.END.strftime("%d/%m/%Y;%H:%M:%S;")+"FUTURE_NOT_A_VALID_NUMBER\n")
    b=io.BytesIO()
    with zipfile.ZipFile(b,"w",zipfile.ZIP_DEFLATED) as z:
        z.writestr("household_power_consumption.txt",out.getvalue())
    return b.getvalue()


class ExternalSourceContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source=archive()

    def test_valid_continuous_fixture(self):
        values,info=s.ten_minute_energy(self.source)
        self.assertEqual(info["missing_power_minute_rows"],0)
        self.assertEqual(len(values),int((s.END-s.START).total_seconds()/600))
        self.assertAlmostEqual(values[0],160.,places=7)

    def test_stress_is_labeled_different_target_not_confirmation(self):
        result=s.compare(self.source)
        self.assertFalse(result["independent_same_task_confirmation"])
        self.assertFalse(result["target_equivalence"])
        self.assertFalse(result["trained_or_calibrated_during_transfer"])
        self.assertGreater(result["complete_scored_examples"],300)
        self.assertGreaterEqual(result["scored_days"],7)

    def test_future_member_rows_not_parsed_for_outcomes(self):
        alt=archive(future_garbage=True)
        vals,_=s.ten_minute_energy(alt)
        self.assertEqual(len(vals),len(s.ten_minute_energy(self.source)[0]))

    def test_missing_block_not_imputed(self):
        changed=archive(missing_block=s.START+timedelta(days=18))
        values,info=s.ten_minute_energy(changed)
        self.assertEqual(info["missing_or_incomplete_ten_minute_blocks"],1)
        self.assertTrue(any(v is None for v in values))

    def test_missing_minute_detected(self):
        changed=archive(skip_minute=s.START+timedelta(days=2,minutes=3))
        with self.assertRaises(s.SourceError):
            s.ten_minute_energy(changed)

    def test_unrecognized_original_header_denied(self):
        changed=archive(bad_header=True)
        with self.assertRaises(s.SourceError):s.ten_minute_energy(changed)

    def test_source_zip_not_arbitrary_bytes(self):
        with self.assertRaises(zipfile.BadZipFile):s.ten_minute_energy(b"some-other-source")

    def test_missing_source_url_no_rewrite(self):
        self.assertTrue(s.URL.startswith("https://archive.ics.uci.edu/static/public/235/"))
        self.assertEqual(s.EVAL_START.isoformat(),"2007-01-01T00:00:00")
        self.assertEqual(s.END.isoformat(),"2007-01-16T00:00:00")

if __name__=="__main__":
    unittest.main()
