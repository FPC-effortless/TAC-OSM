"""Constructed source-shaped REFIT fixtures; no genuine source evaluation."""
import csv
import importlib.util
import io
from pathlib import Path
import sys
import unittest
ROOT=Path(__file__).resolve().parents[1]/"specialist_models"
spec=importlib.util.spec_from_file_location("model01_refit_source_gate",ROOT/"model01_refit_source_gate.py")
m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
def source_fixture(n=152,spacing=8,channel=2,nonmonotonic=False):
    # 152 samples at 8-second cadence cover 1,208 seconds -> two 600s bins
    stream=io.StringIO();writer=csv.writer(stream)
    writer.writerow(m.FIELDS)
    for i in range(n):
        t=1381323600+i*spacing
        if nonmonotonic and i==10:t-=spacing
        vals=["2013-10-09 13:00:00",str(t),"400"]+["100" if j==channel else "0" for j in range(1,10)]
        writer.writerow(vals)
    return stream.getvalue().encode()
class RefitSourceGateFixtures(unittest.TestCase):
    def test_real_like_8_second_sampling_yields_complete_energy(self):
        out=m.qualification("Appliance2",1,source_fixture())
        windows=out["integrated"]["windows"]
        self.assertEqual(len(windows),2)
        self.assertAlmostEqual(windows[0]["energy_wh"],100*600/3600,places=8)
        self.assertFalse(out["target_equivalent_to_UCI374_total_appliances"])
        self.assertFalse(out["integrated"]["original_unobserved_gap_imputation_identifiable"])
    def test_iam_not_aggregate_only(self):
        with self.assertRaises(m.SourceError):
            m.qualification("Aggregate",1,source_fixture())
    def test_outside_refit_house_blocked(self):
        with self.assertRaises(m.SourceError):
            m.qualification("Appliance1",14,source_fixture())
    def test_source_order_rejects_duplicate(self):
        with self.assertRaises(m.SourceError):
            m.qualification("Appliance1",1,source_fixture(nonmonotonic=True))
    def test_large_observation_gap_is_not_invented(self):
        inp=[(600*i,120.) for i in range(3)]
        r=m.integrate_10minute_windows(inp)
        self.assertEqual(r["complete_publisher_cleaned_ten_minute_windows"],0)
        self.assertEqual(r["known_timestamp_gap_seconds_excluded"],1200)
    def test_crossing_bin_boundary_is_integrated_exactly(self):
        r=m.integrate_10minute_windows([(598,360.),(602,360.)])
        self.assertEqual(r["incomplete_windows_excluded"],2)
    def test_nonfinite_power_denied(self):
        src=source_fixture().decode().replace(",100,",",nan,",1)
        with self.assertRaises(m.SourceError):m.read_original_like(src,"Appliance2")
    def test_wrong_csv_schema_denied(self):
        with self.assertRaises(m.SourceError):
            m.read_original_like("Time,Unix,Aggregate,Appliance1\nhello,1,2,3\n","Appliance1")
    def test_only_bounded_bytes(self):
        with self.assertRaises(m.SourceError):m.qualification("Appliance2",1,b" "*25_000_001)
    def test_missing_rows_denied(self):
        with self.assertRaises(m.SourceError):m.read_original_like("Time,Unix,Aggregate\n","Appliance1")
    def test_integrator_rejects_backwards(self):
        with self.assertRaises(m.SourceError):m.integrate_10minute_windows([(5,1.),(4,1.)])
    def test_unknown_metadata_not_claimed(self):
        report=m.qualification("Appliance2",1,source_fixture())
        self.assertFalse(report["operator_source_time_available_at_independently_verified"])
        self.assertTrue(report["publisher_cleaned_gap_fill_ambiguity"])
        self.assertFalse(report["independent_source_predictive_evaluation"])
if __name__=="__main__":unittest.main()
