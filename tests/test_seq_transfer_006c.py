"""SEQ006C pre-test checks. Never generate a reserved TEST example."""
import tempfile
from pathlib import Path
import unittest

from scripts.confirm_seq_transfer_006c import (
    ARMS, EXPECTED_HASHES, FRESH_SEEDS, TEST_SPECS,
    read_data, test_data_after_exposure,
)


class TestConfirmatorySEQ006C(unittest.TestCase):
    def test_all_test_partitions_require_final_marker(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            for name in TEST_SPECS:
                with self.assertRaises(PermissionError):
                    test_data_after_exposure(root, name)
            (root / "TEST-EXPOSURE.json").write_text('{"status":"TEST_EXPOSED_AFTER_60_FITS"}')
            for name in TEST_SPECS:
                with self.assertRaises(PermissionError):
                    test_data_after_exposure(root, name)

    def test_fresh_seed_and_stratum_contract(self):
        self.assertEqual(len(FRESH_SEEDS), 20)
        self.assertEqual(len(set(FRESH_SEEDS)), 20)
        self.assertEqual(tuple(FRESH_SEEDS), tuple(range(690701, 690721)))
        self.assertEqual(set(TEST_SPECS), {"L256_E4", "L512_E4", "L512_E6"})
        self.assertEqual(TEST_SPECS["L512_E6"], (690623, 512, 6))
        self.assertEqual(len(ARMS), 3)

    def test_original_preexposure_data_hashes(self):
        train, validation, manifests = read_data()
        self.assertEqual(len(train[0]), 3000)
        self.assertEqual(len(validation["L64_E4"][0]), 600)
        self.assertEqual(len(validation["L128_E4"][0]), 600)
        self.assertEqual(manifests["train"]["sha256_public_inputs_and_labels"],
                         EXPECTED_HASHES["train"])
        for key in ("L64_E4", "L128_E4"):
            self.assertEqual(manifests["validation"][key]["sha256_public_inputs_and_labels"],
                             EXPECTED_HASHES[key])

if __name__ == "__main__":
    unittest.main()
