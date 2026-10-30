"""Provenance rejection tests and optional constructed no-fit source checks."""
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "research/mathematical-engineering/langmath095"), str(ROOT / "src")]
import checkpoints

try:
    import torch
except ImportError:
    torch = None


class ReceiptFixture(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)
        self.checkpoint = self.folder / "checkpoint.pt"
        self.inputs = self.folder / "inputs.npy"
        self.receipt = self.folder / "receipt.json"
        self.checkpoint.write_bytes(b"not a checkpoint; never load without torch")
        np.save(self.inputs, np.array([[4, 9, 4, 10, 4]], dtype=np.int64), allow_pickle=False)
        self.record = dict(schema="langmath095-checkpoint-v1", model_kind="500k", partition="DEV",
                           publisher="constructed engineering fixture", license="fixture only",
                           checkpoint_sha256=checkpoints.sha256(self.checkpoint),
                           inputs_sha256=checkpoints.sha256(self.inputs),
                           source_commit=checkpoints.BASE_COMMIT,
                           source_sha256=checkpoints.required_source_hashes("500k"))
        self.write_receipt()

    def write_receipt(self):
        self.receipt.write_text(json.dumps(self.record))

    def audit(self):
        return checkpoints.audit_checkpoint(self.checkpoint, self.inputs, self.receipt)


class ProvenanceTests(ReceiptFixture):
    def test_validates_receipt_and_bounded_next_token_array(self):
        record, raw = checkpoints.validate_receipt(self.checkpoint, self.inputs, self.receipt)
        self.assertEqual(record["partition"], "DEV")
        self.assertEqual(raw.dtype, np.int64)
        self.assertEqual(raw.shape, (1, 5))

    def test_invalid_receipts_never_import_or_load_torch(self):
        changes = [dict(partition="TEST"), dict(partition="sealed_TEST"), dict(license=""),
                   dict(publisher=" "), dict(checkpoint_sha256="0"*64), dict(inputs_sha256="0"*64),
                   dict(source_commit="a"*40), dict(source_sha256={}), dict(model_kind="10m"),
                   dict(schema="arbitrary-code")]
        original = self.record.copy()
        for change in changes:
            with self.subTest(change=change):
                self.record = {**original, **change}
                self.write_receipt()
                with mock.patch.object(checkpoints.importlib, "import_module", side_effect=AssertionError("torch import forbidden")):
                    result = self.audit()
                self.assertEqual(result["status"], "BLOCKED")

    def test_missing_torch_returns_inconclusive_after_validation(self):
        with mock.patch.object(checkpoints.importlib, "import_module", side_effect=ModuleNotFoundError("torch unavailable")):
            report = self.audit()
        self.assertEqual(report["status"], "INCONCLUSIVE_TORCH_UNAVAILABLE")
        self.assertEqual(report["optimizer_steps"], 0)
        self.assertIn("not_independently_authenticated", report["provenance"])

    def test_input_ids_shapes_dtypes_and_pickle_fail_closed(self):
        arrays = [np.array([[1, 2]], dtype=np.int32), np.array([1, 2], dtype=np.int64),
                  np.zeros((5, 3), dtype=np.int64), np.zeros((1, 1026), dtype=np.int64),
                  np.array([[1]], dtype=np.int64), np.array([[1, -1]], dtype=np.int64),
                  np.array([[1, 256]], dtype=np.int64), np.array([[object()]], dtype=object)]
        for array in arrays:
            with self.subTest(shape=array.shape, dtype=array.dtype):
                np.save(self.inputs, array, allow_pickle=True)
                self.record["inputs_sha256"] = checkpoints.sha256(self.inputs)
                self.write_receipt()
                with mock.patch.object(checkpoints.importlib, "import_module", side_effect=AssertionError("torch import forbidden")):
                    self.assertEqual(self.audit()["status"], "BLOCKED")

    def test_scan_globals_restore_on_failure(self):
        original_matrix = lambda p, q: "old matrix"
        original_diagonal = lambda p, q: "old diagonal"
        module = SimpleNamespace(affine_scan=original_matrix, scalar_affine_scan=original_diagonal)
        with self.assertRaisesRegex(RuntimeError, "fixture interruption"):
            with checkpoints.scan_substitution("tree", [module]):
                self.assertIsNot(module.affine_scan, original_matrix)
                np.testing.assert_array_equal(module.scalar_affine_scan(
                    np.ones((1, 3, 1)), np.ones((1, 3, 1))), np.array([[[1.], [2.], [3.]]]))
                raise RuntimeError("fixture interruption")
        self.assertIs(module.affine_scan, original_matrix)
        self.assertIs(module.scalar_affine_scan, original_diagonal)


@unittest.skipIf(torch is None, "optional torch dependency unavailable")
class ConstructedCheckpointTests(ReceiptFixture):
    def fixture_model(self):
        torch.manual_seed(95095)
        model = checkpoints._factory("500k").float().eval()
        torch.save({"model": model.state_dict(), "step": 0}, self.checkpoint)
        self.record["checkpoint_sha256"] = checkpoints.sha256(self.checkpoint)
        self.write_receipt()
        return model

    def test_constructed_checkpoint_forward_gradient_and_scan_audit(self):
        self.fixture_model()
        report = self.audit()
        self.assertEqual(report["status"], "PASS_BOUNDED_CPU_FP32_FIXTURES", report.get("reason"))
        self.assertEqual(report["checkpoint_step_metadata_unverified"], 0)
        gradients = report["diagnostics"]["full_named_gradients"]
        self.assertTrue(gradients)
        self.assertTrue(all(x["finite"] is True for x in gradients.values()))
        self.assertTrue(all(x["status"].startswith("PASS") for x in report["scan_source_parity"].values()))
        self.assertEqual(report["full_carried_state_ad"], "OPEN")
        json.dumps(report, allow_nan=False)

    def test_checkpoint_load_is_weights_only_and_strict(self):
        self.fixture_model()
        torch.save({"model": {}}, self.checkpoint)
        self.record["checkpoint_sha256"] = checkpoints.sha256(self.checkpoint)
        self.write_receipt()
        self.assertEqual(self.audit()["status"], "BLOCKED_EXECUTION")

    def test_adapter_uses_real_window_and_explicit_embedding_weight_ad(self):
        model = self.fixture_model()
        adapter = checkpoints.make_window_probe_adapter("500k")
        sample = {"tokens": torch.tensor([[4, 9, 4, 10], [5, 6, 5, 7]])}
        adapter.validate(model, sample)
        self.assertEqual(adapter.steps(sample), 0)
        state = adapter.start(model, sample)
        self.assertIsNone(state)
        self.assertEqual(adapter.state_views(state), {})
        heads, sources = adapter.gradient_graph(model, state, sample)
        self.assertEqual(tuple(heads["logits"].shape), (2, 4, 256))
        self.assertIs(sources["embedding_weight_AD_not_token_input_AD"], model.embedding.weight)
        self.assertEqual(adapter.batch_item(sample, 1)["tokens"].shape, (1, 4))
        with self.assertRaises(ValueError):
            adapter.advance(model, state, sample, 0)


if __name__ == "__main__":
    unittest.main()
