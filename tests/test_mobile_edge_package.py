"""Constructed-unit tests of package *identity*, not a trained mobile inference."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "validate_mobile_edge_package.py"


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class MobileEdgePackageTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        self.root = Path(self.dir.name)
        self.model = b"constructed fixture; not a trained model"
        (self.root / "test.onnx").write_bytes(self.model)
        self.manifest = {
            "schema_version": 1,
            "model": {
                "path": "test.onnx", "sha256": sha(self.model),
                "bytes": len(self.model), "format": "onnx",
                "original_checkpoint_sha256": "a" * 64,
                "source_commit": "fixture-only; NOT a real checkpoint",
                "trainable_parameters": 100_000,
                "deployed_parameters": 100_000,
            },
            "interface": {
                "inputs": [
                    {"name": "observation", "dtype": "float32", "shape": [1, 14]},
                    {"name": "state", "dtype": "float32", "shape": [1, 166]},
                ],
                "outputs": [
                    {"name": "prediction", "dtype": "float32", "shape": [1, 14]},
                    {"name": "next_state", "dtype": "float32", "shape": [1, 166]},
                ],
                "state_pairs": [{"input": "state", "output": "next_state"}],
                "reset_semantics": "all zero only if validated by source model",
            },
            "scientific_evidence": {"status": "FAILED", "evidence_ref": "CONSTRUCTED FIXTURE ONLY"},
        }
        self.write_manifest()

    def write_manifest(self):
        (self.root / "mobile-manifest.json").write_text(json.dumps(self.manifest), encoding="utf-8")

    def run_check(self, *extra):
        run = subprocess.run(
            [sys.executable, str(SCRIPT), str(self.root), *extra],
            capture_output=True, text=True, check=False, timeout=20,
        )
        return run.returncode, json.loads(run.stdout)

    def test_constructed_identity_pass_not_scientific_or_device(self):
        code, result = self.run_check()
        self.assertEqual(code, 0)
        self.assertEqual(result["package_integrity"], "PASS")
        self.assertEqual(result["hardware_evidence"], "BLOCKED_NO_DEVICE_REPORT")
        self.assertEqual(result["scientific_qualification"], "FAILED")
        self.assertEqual(result["deployment_qualification"], "NOT_ESTABLISHED")

    def test_no_device_is_blocking_when_requested(self):
        code, result = self.run_check("--require-device")
        self.assertEqual(code, 2)
        self.assertEqual(result["package_integrity"], "FAIL_OR_BLOCKED")

    def test_tamper_model_rejected(self):
        (self.root / "test.onnx").write_bytes(b"corrupted")
        self.assertEqual(self.run_check()[0], 2)

    def test_path_traversal_rejected(self):
        self.manifest["model"]["path"] = "../other.onnx"
        self.write_manifest()
        self.assertEqual(self.run_check()[0], 2)

    def test_symlink_escape_rejected(self):
        (self.root / "test.onnx").unlink()
        outside = self.root.parent / (self.root.name + "-outside.onnx")
        outside.write_bytes(self.model)
        self.addCleanup(lambda: outside.unlink(missing_ok=True))
        (self.root / "test.onnx").symlink_to(outside)
        self.assertEqual(self.run_check()[0], 2)

    def test_missing_explicit_state_rejected(self):
        self.manifest["interface"]["state_pairs"] = []
        self.write_manifest()
        self.assertEqual(self.run_check()[0], 2)

    def test_out_of_range_neural_scale_rejected(self):
        self.manifest["model"]["trainable_parameters"] = 94_999
        self.write_manifest()
        self.assertEqual(self.run_check()[0], 2)

    def test_no_scientific_status_rejected(self):
        self.manifest["scientific_evidence"]["status"] = "PASS"
        self.write_manifest()
        self.assertEqual(self.run_check()[0], 2)

    def test_real_device_receipt_format_does_not_qualify_model(self):
        receipt = {
            "model_sha256": sha(self.model), "source": "physical-device",
            "device_model": "fixture-labelled-phone", "android_version": "fixture",
            "execution_backend": "fixture", "benchmark_script": "fixture",
            "latency_ms": [float(i + 1) for i in range(30)],
            "peak_rss_bytes": 123456,
        }
        data = json.dumps(receipt).encode()
        (self.root / "device.json").write_bytes(data)
        self.manifest["device_measurement"] = {"path": "device.json", "sha256": sha(data)}
        self.write_manifest()
        code, result = self.run_check("--require-device")
        self.assertEqual(code, 0)
        self.assertEqual(result["hardware_evidence"], "RECEIPT_FORMAT_PASS_INDEPENDENT_AUTHENTICATION_OPEN")
        self.assertEqual(result["deployment_qualification"], "NOT_ESTABLISHED")

    def test_tampered_device_receipt_rejected(self):
        receipt = {"model_sha256": sha(self.model)}
        data = json.dumps(receipt).encode()
        (self.root / "device.json").write_bytes(data)
        self.manifest["device_measurement"] = {"path": "device.json", "sha256": "b" * 64}
        self.write_manifest()
        self.assertEqual(self.run_check("--require-device")[0], 2)


if __name__ == "__main__":
    unittest.main()
