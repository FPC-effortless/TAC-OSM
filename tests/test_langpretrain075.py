"""LANGPRETRAIN075 offline software-only fixtures; no model fit or GPU budget."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from scripts.langpretrain075_audit import AuditError, audit_corpus, cost_plan


class CorpusAuditTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.docs = []
        for category, split, text in [
            ("web", "FIT", "public educational prose " * 10),
            ("code", "FIT", "print('source-only') # coding\n" * 10),
            ("technical", "FIT", "a verified manual describes an operation\n" * 6),
            ("dialogue", "FIT", "person A: hello\nperson B: hello again\n" * 9),
            ("web", "DEV", "held-out independent prose in a development file\n" * 8),
        ]:
            idx = len(self.docs)
            name = f"original-{idx}.txt"
            raw = text.encode("utf-8")
            (self.root / name).write_bytes(raw)
            self.docs.append({
                "id": f"publisher-doc-{idx}",
                "path": name, "split": split, "category": category,
                "sha256": hashlib.sha256(raw).hexdigest(),
                "license": "fixture-permissive",
                "publisher": "constructed-test-fixture",
                "source_url": "https://example.org/fixture",
            })
        self.manifest = self.root / "manifest.json"
        self.write_manifest()

    def write_manifest(self):
        self.manifest.write_text(json.dumps({
            "schema": "tac-osm.byte-pretrain-corpus.v1",
            "documents": self.docs,
        }), encoding="utf-8")

    def test_valid_fixture_audits_bytes_and_document_boundaries(self):
        report = audit_corpus(self.manifest, self.root)
        self.assertEqual(report["status"], "SOURCE_DIGEST_AND_UTF8_CHECKED_NOT_AUTHORIZED")
        self.assertEqual(report["gpu_minutes_consumed"], 0)
        self.assertFalse(report["approved_to_train"])
        self.assertEqual(len(report["documents"]), 5)
        self.assertEqual(report["within_document_windows"]["FIT"],
                         sum(len((self.root / d["path"]).read_bytes()) - 128
                             for d in self.docs if d["split"] == "FIT"))
        self.assertIn("near_dedup", report["checks_not_performed"])
        self.assertEqual(report["protected_TEST"], "NOT_ACCESSED")

    def test_rejects_mutated_original_bytes(self):
        (self.root / self.docs[0]["path"]).write_text("tampered after registration")
        with self.assertRaisesRegex(AuditError, "digest mismatch"):
            audit_corpus(self.manifest, self.root)

    def test_rejects_cross_partition_duplicates(self):
        self.docs[-1]["path"] = self.docs[0]["path"]
        self.docs[-1]["sha256"] = self.docs[0]["sha256"]
        self.write_manifest()
        with self.assertRaisesRegex(AuditError, "duplicate source"):
            audit_corpus(self.manifest, self.root)

    def test_rejects_test_partition(self):
        self.docs[-1]["split"] = "TEST"
        self.write_manifest()
        with self.assertRaisesRegex(AuditError, "TEST"):
            audit_corpus(self.manifest, self.root)

    def test_rejects_path_traversal(self):
        self.docs[0]["path"] = "../private.txt"
        self.write_manifest()
        with self.assertRaisesRegex(AuditError, "traversal"):
            audit_corpus(self.manifest, self.root)

    def test_rejects_symlinks(self):
        (self.root / "link.txt").symlink_to(self.root / self.docs[0]["path"])
        self.docs[0]["path"] = "link.txt"
        self.write_manifest()
        with self.assertRaisesRegex(AuditError, "symlink"):
            audit_corpus(self.manifest, self.root)

    def test_rejects_invalid_utf8(self):
        path = self.root / self.docs[0]["path"]
        raw = b"\xff" + b"x" * 150
        path.write_bytes(raw)
        self.docs[0]["sha256"] = hashlib.sha256(raw).hexdigest()
        self.write_manifest()
        with self.assertRaises(UnicodeError):
            audit_corpus(self.manifest, self.root)

    def test_cost_is_unknown_without_measured_gpu_rate(self):
        report = cost_plan(10_000_000, None, 0.0)
        self.assertIsNone(report["estimated_paid_compute_per_seed_usd"])
        self.assertFalse(report["approved_to_spend"])
        self.assertGreater(report["cpu_extrapolated_hours_PER_SEED_NOT_FORECAST"], 0)

    def test_cost_accounts_for_two_concurrent_seed_shards(self):
        # Aggregate rate 20k byte targets/sec across BOTH physical T4s.
        report = cost_plan(10_000_000, 20_000.0, 1.0)
        self.assertEqual(report["gpu_count"], 2)
        self.assertAlmostEqual(report["gpu_wall_hours_PER_SEED"], 1000 / 3600)
        self.assertAlmostEqual(report["estimated_paid_compute_per_seed_usd"], 2000 / 3600)
        with self.assertRaises(AuditError):
            cost_plan(10_000_000, float("nan"), 0.0)
        with self.assertRaises(AuditError):
            cost_plan(10_000_000, 20_000, -1.0)


if __name__ == "__main__":
    unittest.main()
