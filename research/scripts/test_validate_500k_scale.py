"""New-only validation and historical immutability contract."""
import unittest
from validate_500k_scale import validate_study_counts, TARGET, MIN_ACTIVE, MAX_ACTIVE

class Scale500K(unittest.TestCase):
    def test_target_and_band(self):
        r=validate_study_counts({"TAC":500_000,"GRU":499_100,"SSM":500_600},
                                study_date="2026-10-28")
        self.assertEqual(r.target,TARGET)
        self.assertEqual((r.lower,r.upper),(475000,525000))
    def test_band_endpoints(self):
        self.assertEqual(validate_study_counts({"a":MIN_ACTIVE},
                                               study_date="2026-10-29").target,TARGET)
        self.assertEqual(validate_study_counts({"b":MAX_ACTIVE},
                                               study_date="2026-10-29").upper,MAX_ACTIVE)
    def test_legacy_not_reinterpreted(self):
        with self.assertRaises(ValueError):
            validate_study_counts({"historical_native":100_680},
                                  study_date="2026-10-27")
    def test_old_100k_rejected_when_new(self):
        with self.assertRaises(ValueError):
            validate_study_counts({"new":100_680},study_date="2026-10-28")
    def test_one_percent_parity(self):
        with self.assertRaises(ValueError):
            validate_study_counts({"TAC":500_000,"baseline":510_000},
                                  study_date="2026-10-28")
    def test_no_faux_counts(self):
        for bad in (500_000.0,True,"500000",None,470_000,530_000):
            with self.subTest(bad=bad),self.assertRaises(ValueError):
                validate_study_counts({"bad":bad},study_date="2026-10-28")
if __name__ == "__main__":
    unittest.main()
