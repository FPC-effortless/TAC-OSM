"""SEQ009 ingestion guard tests; generated tiny CSV fixtures ONLY for code correctness.

No synthetic-model accuracy is claimed or measured.
"""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from scripts.acquire_seq009_scania_zip import ALLOW_FILES, ARCHIVE_URL, extract_approved
from scripts.audit_seq009_scania_source import (
    FILES, DOI, audit, audit_csv,
)

class TestRealSourcePreflight(unittest.TestCase):
    def test_archive_allowlist_blocks_test_members(self):
        self.assertIn("/2024-34/3/file/zip", ARCHIVE_URL)
        self.assertEqual(set(FILES), set(ALLOW_FILES))
        self.assertNotIn("test_labels.csv", ALLOW_FILES)

    def fixture(self, root):
        docs = {
            "train_tte.csv":
                "vehicle_id,length_of_study_time_step,in_study_repair\n"
                "a,120,1\nb,150,0\n",
            "train_specifications.csv":
                "vehicle_id,spec1\na,Cat1\nb,Cat2\n",
            "validation_labels.csv":
                "vehicle_id,class_label\nc,0\nd,3\n",
        }
        target = root / "raw"
        target.mkdir()
        manifest = {"dataset_doi": DOI, "version": 3, "files": {}}
        for filename, content in docs.items():
            data = content.encode()
            (target / filename).write_bytes(data)
            manifest["files"][filename] = {"sha256": hashlib.sha256(data).hexdigest()}
        return target, manifest

    def test_reference_fixture_contract(self):
        with tempfile.TemporaryDirectory() as td:
            raw, manifest = self.fixture(Path(td))
            result = audit(raw, manifest)
            self.assertEqual(result["files"]["train_tte.csv"]["repaired"], 1)
            self.assertEqual(result["files"]["train_tte.csv"]["censored"], 1)
            self.assertEqual(result["files"]["validation_labels.csv"]["class_counts"], {"0":1,"3":1})
            self.assertFalse(result["test_labels_opened"])
            self.assertEqual(result["models_fitted"], 0)

    def test_wrong_hash_and_identity_overlap_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            root, manifest = self.fixture(Path(td))
            manifest["files"]["train_tte.csv"]["sha256"] = "0" * 64
            with self.assertRaisesRegex(ValueError, "altered"):
                audit(root, manifest)
            manifest["files"]["train_tte.csv"]["sha256"] = hashlib.sha256(
                (root / "train_tte.csv").read_bytes()).hexdigest()
            (root / "validation_labels.csv").write_text("vehicle_id,class_label\na,0\nd,3\n")
            manifest["files"]["validation_labels.csv"]["sha256"] = hashlib.sha256(
                (root / "validation_labels.csv").read_bytes()).hexdigest()
            with self.assertRaisesRegex(ValueError, "overlap"):
                audit(root, manifest)

    def test_schema_mismatch_and_duplicate_identity_fail(self):
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/"train_tte.csv"
            path.write_text("vehicle_id,in_study_repair\na,1\n")
            with self.assertRaisesRegex(ValueError, "Schema mismatch"):
                audit_csv(path, "train_tte.csv")
            path.write_text(
                "vehicle_id,length_of_study_time_step,in_study_repair\n"
                "a,100,1\na,120,0\n"
            )
            with self.assertRaisesRegex(ValueError, "duplicate"):
                audit_csv(path, "train_tte.csv")


if __name__ == "__main__":
    unittest.main()
