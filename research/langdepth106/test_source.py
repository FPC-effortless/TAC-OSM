"""Constructed source/split integrity fixtures; no optimizer or model scoring."""

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from research.langdepth106.source import (
    FORMAT, COMMIT, GIT_BLOB_SHA1, SOURCE_BYTES, SOURCE_SHA256,
    git_blob_sha1, partition_ranges, sha256, source_target_window,
    split_raw_bytes, verify_original, verify_partition, verify_training_stage,
)


class SourceIntegrityTests(unittest.TestCase):
    def test_git_blob_hash_includes_type_and_length(self):
        self.assertEqual(git_blob_sha1(b"hello\n"), "ce013625030ba8dba906f756967f9e9ca394464a")
        self.assertNotEqual(git_blob_sha1(b"hello\n"), hashlib.sha1(b"hello\n").hexdigest())

    def test_integer_ranges_and_exact_reconstruction(self):
        raw = bytes(range(200))
        parts, meta = split_raw_bytes(raw)
        self.assertEqual(partition_ranges(200), {"train": (0, 180), "dev": (180, 190), "test": (190, 200)})
        self.assertEqual(b"".join(parts.values()), raw)
        for name, part in parts.items():
            self.assertEqual(meta[name]["sha256"], sha256(part))
            self.assertEqual(meta[name]["adjacent_source_target_pairs"], len(part) - 1)
        self.assertEqual(sum(m["adjacent_source_target_pairs"] for m in meta.values()), len(raw) - 3)
        self.assertEqual(partition_ranges(SOURCE_BYTES), {"train": (0, 1003854), "dev": (1003854, 1059624), "test": (1059624, 1115394)})

    def test_no_source_target_window_crosses_partition(self):
        parts, _ = split_raw_bytes(bytes(range(200)))
        for part in parts.values():
            x, y = source_target_window(part, len(part) - 5, 4)
            self.assertEqual(x[1:], y[:-1])
            self.assertEqual(y[-1], part[-1])
            with self.assertRaises(ValueError):
                source_target_window(part, len(part) - 4, 4)
        with self.assertRaises(ValueError):
            source_target_window(parts["train"], -1, 4)
        with self.assertRaises(ValueError):
            source_target_window(parts["train"], 0, True)

    def test_mutations_outside_train_cannot_change_train_bytes(self):
        original = bytes(range(200))
        parts, _ = split_raw_bytes(original)
        poisoned = original[:180] + b"!" * 20
        poisoned_parts, _ = split_raw_bytes(poisoned)
        self.assertEqual(parts["train"], poisoned_parts["train"])
        self.assertNotEqual(parts["dev"], poisoned_parts["dev"])
        self.assertNotEqual(parts["test"], poisoned_parts["test"])

    def test_wrong_original_is_rejected(self):
        for raw in (b"wrong original", b"x" * SOURCE_BYTES):
            with self.assertRaises(ValueError):
                verify_original(raw)
        with self.assertRaises(TypeError):
            verify_original("text is not raw bytes")

    def test_invalid_split_inputs_rejected(self):
        for value in (-1, True, 39, 200.0):
            with self.assertRaises(ValueError):
                partition_ranges(value)
        with self.assertRaises(TypeError):
            split_raw_bytes(bytearray(b"x" * 100))

    def test_training_stage_rejects_forged_receipt_and_extra_eval_files(self):
        # Even a self-consistent rewritten receipt cannot substitute fixture bytes
        # for the frozen real TRAIN partition. No source-authenticity PASS is
        # inferred from this constructed negative control.
        with tempfile.TemporaryDirectory() as tmp:
            stage = Path(tmp)
            raw = b"x" * 1003854
            receipt = {"format": FORMAT, "stage_access": "TRAIN_ONLY", "source": {
                "sha256": SOURCE_SHA256, "git_blob_sha1": GIT_BLOB_SHA1, "commit": COMMIT,
            }, "partitions": {"train": {
                "bytes": len(raw), "sha256": sha256(raw),
                "source_byte_start_inclusive": 0, "source_byte_end_exclusive": len(raw),
            }}}
            (stage / "train.bin").write_bytes(raw)
            (stage / "source_receipt.json").write_text(json.dumps(receipt), encoding="utf-8")
            with self.assertRaises(ValueError):
                verify_training_stage(stage)
            (stage / "dev.bin").write_bytes(b"no eval bytes in fit staging")
            with self.assertRaises(ValueError):
                verify_training_stage(stage)
            (stage / "dev.bin").unlink()
            (stage / "train.bin").write_bytes(b"y" + raw[1:])
            with self.assertRaises(ValueError):
                verify_training_stage(stage)

    def test_partition_hash_is_independent_of_receipt(self):
        for name, size in (("train", 1003854), ("dev", 55770), ("test", 55770)):
            with self.assertRaises(ValueError):
                verify_partition(b"x" * size, name)
        with self.assertRaises(ValueError):
            verify_partition(b"", "unknown")

    def test_training_stage_rejects_symlink(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            (path / "target").write_bytes(b"data")
            stage = path / "stage"
            stage.mkdir()
            (stage / "train.bin").symlink_to(path / "target")
            (stage / "source_receipt.json").write_text("{}", encoding="utf-8")
            with self.assertRaises(ValueError):
                verify_training_stage(stage)


if __name__ == "__main__":
    unittest.main()
