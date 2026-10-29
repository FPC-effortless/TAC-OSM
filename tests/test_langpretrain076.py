"""LANGPRETRAIN076 deterministic no-fit policy tests. Does NOT simulate a GPU score."""
import importlib.util
import math
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "langpretrain076_t4_measure", ROOT / "scripts/langpretrain076_t4_measure.py")
worker = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(worker)


class LangPretrain076Tests(unittest.TestCase):
    def test_frozen_original_training_geometry(self):
        self.assertEqual(worker.PARAMETERS, 499623)
        self.assertEqual(worker.SEEDS, (1701, 1702))
        self.assertEqual(worker.WARMUP_STEPS, 8)
        self.assertEqual(worker.TIMED_STEPS, 32)
        self.assertEqual(worker.TRAINING_BYTES_PER_SEED, 40960)
        self.assertEqual(worker.TIMED_BYTES_PER_SEED, 32768)
        self.assertLessEqual(worker.MAX_WORKER_SECONDS, 600)

    def test_timed_rate_is_original_byte_targets_not_bpe(self):
        self.assertAlmostEqual(worker.timed_rate(32768, 4.0), 8192.0)
        for bad in (0, -1, float("nan"), float("inf")):
            with self.assertRaises(worker.ThroughputGateError):
                worker.timed_rate(32768, bad)
        with self.assertRaises(worker.ThroughputGateError):
            worker.timed_rate(-1, 1.)

    def test_hashes_and_bundle_needed_before_gpu(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises((OSError, worker.ThroughputGateError)):
                worker.preflight_source(Path(d))

    def test_source_model_frozen_digest_is_64_lowercase_hex(self):
        for value in (worker.CONTEXT_SHA, worker.SOURCE_SHA):
            self.assertEqual(len(value),64)
            self.assertTrue(all(c in "0123456789abcdef" for c in value))
        self.assertEqual(len(worker.UPPER_GIT_BLOB),40)

    def test_no_gpu_or_paid_cloud_execution_in_software_tests(self):
        self.assertFalse(hasattr(worker, "dispatch_kaggle"))
        self.assertNotIn("TEST", str(worker.SEEDS))


if __name__ == "__main__":
    unittest.main()
