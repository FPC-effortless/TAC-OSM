"""Constructed-only provenance and REFIT 8-second period integration diagnostics."""
from datetime import datetime,timezone,timedelta
import importlib.util
from pathlib import Path
import sys
import unittest
ROOT=Path(__file__).resolve().parents[1]/"specialist_models"
spec=importlib.util.spec_from_file_location("model01_refit_metadata_and_target_gate",
                                            ROOT/"model01_refit_metadata_and_target_gate.py")
m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)

def metadata():
    return {"id":5063428,
            "metadata":{"doi":m.DOI,"title":"REFIT: Electrical Load Measurements (Cleaned)",
                        "license":{"id":"cc-by-4.0"}},
            "files":[{"key":"CLEAN_House1.csv","size":400000000,
                      "checksum":"md5:abcdefghabcdefghabcdefghabcdefgh"}]}

def sequence():
    start=datetime(2014,1,1,tzinfo=timezone.utc)
    return [{"at_utc":(start+timedelta(seconds=8*i)).isoformat().replace("+00:00","Z"),
             "watts":120.0,"actually_observed":True} for i in range(75)]

class RefitContract(unittest.TestCase):
    def test_genuine_record_schema_and_target_guard(self):
        x=m.validate_record(metadata())
        self.assertEqual(x["license_metadata_check"],"METADATA_CC_BY_4_0")
        self.assertFalse(x["target_equivalence_to_uci374"])
        self.assertFalse(x["model_fitted_or_scored"])
    def test_missing_license_not_fabricated(self):
        j=metadata();del j["metadata"]["license"]
        self.assertEqual(m.validate_record(j)["license_metadata_check"],
                         "LICENSE_FIELD_NOT_MACHINE_VERIFIED")
    def test_wrong_doi_blocked(self):
        j=metadata();j["metadata"]["doi"]="10.5281/zenodo.0"
        with self.assertRaises(m.SourceError):m.validate_record(j)
    def test_no_source_file_blocked(self):
        j=metadata();j["files"]=[]
        with self.assertRaises(m.SourceError):m.validate_record(j)
    def test_75_8second_samples_integrate_watts_to_wh(self):
        self.assertAlmostEqual(m.checked_wh(sequence()),20.,places=10)
    def test_missing_source_point_forbidden(self):
        seq=sequence();seq[4]["actually_observed"]=False
        with self.assertRaises(m.SourceError):m.checked_wh(seq)
    def test_gap_rejected_not_forward_filled(self):
        seq=sequence();seq[23]["at_utc"]=seq[22]["at_utc"]
        with self.assertRaises(m.SourceError):m.checked_wh(seq)
    def test_fewer_samples_rejected(self):
        with self.assertRaises(m.SourceError):m.checked_wh(sequence()[:-1])
    def test_invalid_negative_imputed_value_rejected(self):
        seq=sequence();seq[2]["watts"]=-1.
        with self.assertRaises(m.SourceError):m.checked_wh(seq)
    def test_true_is_not_numerical_watt(self):
        seq=sequence();seq[2]["watts"]=True
        with self.assertRaises(m.SourceError):m.checked_wh(seq)
    def test_identity_and_source_appliance_not_implied_by_aggregate(self):
        x=m.validate_record(metadata())
        self.assertFalse(x["source_reconstructed_targets"])
        self.assertFalse(x["refit_cleaned_missingness_native"])

if __name__=="__main__":
    unittest.main()
