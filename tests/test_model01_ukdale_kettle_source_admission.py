"""Constructed unit and source-gate tests. Authentic publisher metadata checked in CI."""
import hashlib
import io
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"specialist_models"))
import model01_ukdale_kettle_source_admission as uk

START=1356998400

def publisher_like():
    dataset={"name":"UK-DALE","number_of_buildings":5,
             "rights_list":[{"name":"Creative Commons Attribution 4.0 International (CC BY 4.0)"}]}
    b={}
    for house,meter in ((1,10),(2,8),(3,2),(4,3),(5,18)):
        appliances=[{"type":"kettle","meters":[meter],"instance":1}]
        if house in (1,4):
            appliances.append({"type":"food processor","meters":[meter],"instance":1})
        b[house]={"instance":house,"appliances":appliances,
                  "elec_meters":{meter:{"data_location":f"house_{house}/channel_{meter}.dat",
                     "device_model":"EcoManagerTxPlug","preprocessing_applied":{"clip":{"upper_limit":4000}}}}}
    return dataset,b

def metadata():
    a,b=publisher_like()
    return uk.verify_metadata(a,b)

def watts_stream(n=200,start=START):
    return b"".join(f"{start+6*i} {35+(i%7)}\n".encode() for i in range(n))

class UKDALEAdmission(unittest.TestCase):
    def test_exact_three_single_kettle_site_metadata_and_roles(self):
        d=metadata()
        self.assertEqual(set(d["eligible_source_meters"]),{"2","3","5"})
        self.assertEqual(d["eligible_source_meters"]["2"]["meter_channel"],8)
        self.assertEqual(d["eligible_source_meters"]["3"]["meter_channel"],2)
        self.assertEqual(d["eligible_source_meters"]["5"]["meter_channel"],18)
        self.assertEqual(d["excluded_meters"]["1"]["reason"],"KETTLE_SHARED_WITH_OTHER_DEVICES")
        self.assertEqual(d["excluded_meters"]["4"]["reason"],"KETTLE_SHARED_WITH_OTHER_DEVICES")
        self.assertFalse(d["original_UCI374_Appliances_target_equivalent"])
        self.assertFalse(d["holdout_house5_power_rows_opened"])
    def test_composite_ineligible_house_must_remain_excluded(self):
        a,b=publisher_like()
        b[4]["appliances"]=b[4]["appliances"][:1]
        with self.assertRaises(uk.AdmissionError):uk.verify_metadata(a,b)
    def test_dedicated_test_meter_cannot_be_shared(self):
        a,b=publisher_like()
        b[5]["appliances"].append({"type":"lamp","meters":[18]})
        with self.assertRaises(uk.AdmissionError):uk.verify_metadata(a,b)
    def test_invalid_meter_identity_rejected(self):
        a,b=publisher_like()
        b[2]["elec_meters"][8]["device_model"]="EcoManagerWholeHouseTx"
        with self.assertRaises(uk.AdmissionError):uk.verify_metadata(a,b)
    def test_metadata_rights_fail_closed(self):
        a,b=publisher_like()
        a["rights_list"]=[]
        with self.assertRaises(uk.AdmissionError):uk.verify_metadata(a,b)
    def test_original_logged_watts_not_fabricated_wh(self):
        raw=watts_stream()
        d=uk.aggregate_kettle_snapshots(io.BytesIO(raw),2,metadata(),hashlib.sha256(raw).hexdigest())
        self.assertEqual(d["accepted_complete_10min_windows"],2)
        self.assertTrue(all(x["actual_source_log_sample_count"]==100 for x in d["recorded_windows"]))
        self.assertEqual(d["original_raw_rows"],200)
        self.assertFalse(d["source_sensor_available_at_known"])
        self.assertIn("sample",d["target"])
        self.assertNotIn("source_time_qualified",d)
    def test_source_hash_mismatch_rejected(self):
        raw=watts_stream()
        with self.assertRaises(uk.AdmissionError):
            uk.aggregate_kettle_snapshots(io.BytesIO(raw),2,metadata(),"0"*64)
    def test_house5_data_access_blocked_before_read(self):
        raw=watts_stream()
        with self.assertRaises(uk.AdmissionError):
            uk.aggregate_kettle_snapshots(io.BytesIO(raw),5,metadata(),hashlib.sha256(raw).hexdigest())
    def test_duplicate_timestamp_rejected(self):
        raw=watts_stream(100)
        raw+=f"{START+594} 70\n".encode()
        with self.assertRaises(uk.AdmissionError):
            uk.aggregate_kettle_snapshots(io.BytesIO(raw),3,metadata(),hashlib.sha256(raw).hexdigest())
    def test_malformed_unit_or_negative_watts_rejected(self):
        for raw in [b"1356998400 -1\n",b"1356998400 17.3\n",b"1356998400 5000\n"]:
            with self.assertRaises(uk.AdmissionError):
                uk.aggregate_kettle_snapshots(io.BytesIO(raw),2,metadata(),hashlib.sha256(raw).hexdigest())
    def test_missing_logger_samples_not_filled(self):
        raw=b"".join(f"{START+6*i} 20\n".encode() for i in range(80))
        with self.assertRaises(uk.AdmissionError):
            uk.aggregate_kettle_snapshots(io.BytesIO(raw),2,metadata(),hashlib.sha256(raw).hexdigest())
    def test_gap_invalidates_window(self):
        # Remove 20 readings: timestamp gap 126s. Neither 10-min bucket is valid.
        raw=b"".join(f"{START+6*i} 20\n".encode() for i in range(100) if not 20<=i<40)
        with self.assertRaises(uk.AdmissionError):
            uk.aggregate_kettle_snapshots(io.BytesIO(raw),2,metadata(),hashlib.sha256(raw).hexdigest())
    def test_source_metadata_requires_exact_original_revision(self):
        d=metadata()
        d["metadata_commit"]="not original"
        raw=watts_stream()
        with self.assertRaises(uk.AdmissionError):
            uk.aggregate_kettle_snapshots(io.BytesIO(raw),2,d,hashlib.sha256(raw).hexdigest())
    def test_no_raw_publisher_archive_fetch_from_audit_mode(self):
        self.assertEqual(uk.ROLES[5],"SEALED_TEST_METADATA_ONLY")
        self.assertEqual(uk.RELEASE_DOI,"10.5286/UKERC.EDC.000004")
        self.assertEqual(uk.MIN_SAMPLES,90)

if __name__=="__main__":
    unittest.main()
