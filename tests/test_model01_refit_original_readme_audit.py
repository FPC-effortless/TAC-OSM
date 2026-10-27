"""Constructed original README contractual claims; no model skill evidence."""
import importlib.util
from pathlib import Path
import sys
import unittest
ROOT=Path(__file__).resolve().parents[1]/"specialist_models"
spec=importlib.util.spec_from_file_location("model01_refit_original_readme_audit",ROOT/"model01_refit_original_readme_audit.py")
m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)

def readme_text():
    return (("REFIT: Electrical Load Measurements INFORMATION 2015\n"
             "LICENCING This work is licensed under the Creative Commons Attribution 4.0 International Public License.\n"
             "House 1 has monitored appliances. "
             "Additionally data was only recorded when there was a change in load; "
             "this data has been filled with intermediate values where not available. "
             "The sensors are also not synchronised as our collection script polled every 6-8 seconds; "
             "the sensor may have updated anywhere in the last 6 seconds. "
             "Occupants can change appliances: we cannot guarantee this to be the case. ")*3).encode()

class PublisherReadmeTests(unittest.TestCase):
    def test_risk_flags_are_truthful(self):
        result=m.inspect(readme_text())
        self.assertFalse(result["eight_second_interval_represents_independent_observation"])
        self.assertFalse(result["native_8second_actually_observed_mask_in_plain_CSV"])
        self.assertFalse(result["per_appliance_identity_unchanged_proven"])
        self.assertFalse(result["model_fitted_or_evaluated"])
        self.assertEqual(result["scientific_qualification"],"BLOCKED")
    def test_sha_matches_exact_original_bytes(self):
        import hashlib
        result=m.inspect(readme_text())
        self.assertEqual(result["original_unmodified_README_sha256"],
                         hashlib.sha256(readme_text()).hexdigest())
    def test_removed_missingness_admission_blocks(self):
        with self.assertRaises(m.ProvenanceError):
            m.inspect(readme_text().replace(b"filled with intermediate values",b"measured 8s samples"))
    def test_removed_iam_identity_warning_blocks(self):
        with self.assertRaises(m.ProvenanceError):
            m.inspect(readme_text().replace(b"we cannot guarantee this to be the case",b"identity stable"))
    def test_bad_publisher_source_input_blocks(self):
        for v in (b"",b"raw-source",None):
            with self.assertRaises(m.ProvenanceError):
                m.inspect(v)
    def test_original_source_version_not_cleaned_Zenodo(self):
        self.assertTrue(m.DOI.startswith("10.15129/"))
        self.assertIn("REFITREADME.txt",m.URL)
        self.assertNotEqual(m.DOI,"10.5281/zenodo.5063428")

if __name__=="__main__":
    unittest.main()
