"""Source-only registration/dispatch tests. No fits or approval evidence.

Mock metadata fixtures are deliberately not valid run grants. These checks
exercise static registration wiring, not runtime ordering or user authorship.
"""
import ast
import contextlib
import io
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

from scripts import validate_research_parity as gate

_probe_spec = importlib.util.spec_from_file_location(
    "synthetic_no_torch_probe", Path(__file__).with_name("synthetic_no_torch_probe.py"))
no_torch_probe = importlib.util.module_from_spec(_probe_spec)
_probe_spec.loader.exec_module(no_torch_probe)


class SyntheticRegistrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.study = "bounded-screen-v1"
        self.relative = "research/synthetic_diagnostics/" + self.study
        directory = self.root / self.relative
        directory.mkdir(parents=True)
        self.manifest = {"study_type": "synthetic_diagnostic_fit",
                         "runner_path": self.relative + "/runner.py",
                         "admission_path": self.relative + "/admission.py"}
        self.manifest_path = directory / "FAIR_COMPARISON.json"
        self.manifest_path.write_text(json.dumps(self.manifest))
        (directory / "admission.py").write_text(
            "def enforce(m, **evidence):\n"
            "    return policy.validate_synthetic_diagnostic(m, 'preflight', **evidence)\n"
            "def final(m, **evidence):\n"
            "    return policy.validate_synthetic_diagnostic(m, phase='final', **evidence)\n")
        (directory / "runner.py").write_text(
            "import admission\n"
            "def run():\n"
            "    admission.enforce(manifest)\n"
            "    import torch\n"
            "    admission.final(manifest, actual=receipt)\n")

    def check(self):
        metadata = {"status": "PASS_SYNTHETIC_REGISTRATION_ONLY_NOT_AUTHORIZED"}
        fake = types.SimpleNamespace(validate_scope=lambda m: metadata)
        with patch.object(gate, "find_changed_synthetic_studies", return_value=[self.study]), \
             patch.object(gate, "_synthetic_policy", return_value=fake), \
             contextlib.redirect_stdout(io.StringIO()) as output:
            result = gate.check_changed_synthetic("BASE...HEAD", self.root)
        return result, json.loads(output.getvalue())

    def test_valid_registration_remains_explicitly_not_authorized(self):
        result, output = self.check()
        self.assertEqual(result, ["synthetic_diagnostics/" + self.study])
        self.assertIs(output["execution_authorized"], False)
        self.assertIn("NOT_AUTHORIZED", output["registration_only"]["status"])
        no_torch_probe.assert_isolated_no_torch(self, __file__)

    def test_missing_root_manifest_fails(self):
        self.manifest_path.unlink()
        with self.assertRaisesRegex(gate.ParityError, "root FAIR_COMPARISON"):
            self.check()

    def test_declared_paths_cannot_escape_repository(self):
        for unsafe in ("../admission.py", "/tmp/admission.py", "a/./admission.py", "a\\admission.py"):
            with self.subTest(path=unsafe):
                self.manifest["admission_path"] = unsafe
                self.manifest_path.write_text(json.dumps(self.manifest))
                with self.assertRaisesRegex(gate.ParityError, "repository-relative"):
                    self.check()

    def test_symlink_cannot_escape_repository(self):
        with tempfile.TemporaryDirectory() as outside:
            external = Path(outside) / "gate.py"
            external.write_text("pass\n")
            admission = self.root / self.manifest["admission_path"]
            admission.unlink()
            admission.symlink_to(external)
            with self.assertRaisesRegex(gate.ParityError, "within the repository"):
                self.check()

    def test_non_source_comments_are_not_gate_hooks(self):
        admission = self.root / self.manifest["admission_path"]
        admission.write_text(
            "# validate_synthetic_diagnostic(m, 'preflight')\n"
            "# validate_synthetic_diagnostic(m, 'final')\n")
        with self.assertRaisesRegex(gate.ParityError, "preflight and final"):
            self.check()

    def test_missing_final_or_early_torch_rejects(self):
        runner = self.root / self.manifest["runner_path"]
        runner.write_text("import admission\nadmission.enforce(manifest)\n")
        with self.assertRaisesRegex(gate.ParityError, "enforce and admission.final"):
            self.check()
        runner.write_text("import torch\nimport admission\nadmission.enforce(manifest)\nadmission.final(manifest)\n")
        with self.assertRaisesRegex(gate.ParityError, "before Torch"):
            self.check()

    def test_invalid_python_is_a_closed_registration_failure(self):
        (self.root / self.manifest["runner_path"]).write_text("def\n")
        with self.assertRaisesRegex(gate.ParityError, "valid Python source"):
            self.check()

    def test_scope_validation_is_not_skipped(self):
        fake = types.SimpleNamespace(validate_scope=lambda m: (_ for _ in ()).throw(ValueError("wrong scope")))
        with patch.object(gate, "find_changed_synthetic_studies", return_value=[self.study]), \
             patch.object(gate, "_synthetic_policy", return_value=fake):
            with self.assertRaisesRegex(gate.ParityError, "wrong scope"):
                gate.check_changed_synthetic("BASE...HEAD", self.root)

    def test_synthetic_discovery_does_not_reclassify_me_dev_paths(self):
        completed = subprocess.CompletedProcess([], 0, stdout=(
            "research/mathematical-engineering/me073/code.py\n"
            "research/synthetic_diagnostics/bounded-screen-v1/code/runner.py\n"
            "research/synthetic_diagnostics/bounded-screen-v1/FAIR_COMPARISON.json\n"
            "scripts/dev062.py\n"))
        with patch.object(gate.subprocess, "run", return_value=completed):
            self.assertEqual(gate.find_changed_studies("BASE...HEAD"), ["dev062", "me073"])
            self.assertEqual(gate.find_changed_synthetic_studies("BASE...HEAD"), [self.study])

    def test_dispatch_forwards_external_evidence_without_deriving_trust(self):
        recorded = []
        def validate(*args, **kwargs):
            recorded.append((args, kwargs))
            return {"status": "CONSTRUCTED_DISPATCH_ONLY", "qualified": False}
        fake = types.SimpleNamespace(validate_synthetic_diagnostic=validate)
        with patch.object(gate, "_synthetic_policy", return_value=fake), \
             patch.object(gate, "_check_manifest_parity_only", side_effect=AssertionError("old path used")):
            result = gate.check_manifest(self.manifest, "final", approval_bytes=b"approval",
                grant_bytes=b"grant", review_bytes=b"review", trusted_approval_sha256="a" * 64,
                trusted_grant_sha256="b" * 64, actual={"fixture": True})
        self.assertFalse(result["qualified"])
        self.assertEqual(recorded[0][0], (self.manifest, "final"))
        self.assertEqual(recorded[0][1]["trusted_approval_sha256"], "a" * 64)
        self.assertEqual(recorded[0][1]["trusted_grant_sha256"], "b" * 64)
        self.assertEqual(recorded[0][1]["actual"], {"fixture": True})


    def test_cli_forwards_exact_bytes_and_external_anchors(self):
        files = {}
        for name in ("approval", "review", "grant"):
            files[name] = self.root / (name + ".json")
            files[name].write_bytes((" { \"fixture\": \"" + name + "\" } \n").encode())
        actual = self.root / "actual.json"
        actual.write_text('{"fixture": true}')
        argv = ["validate_research_parity.py", "--manifest", str(self.manifest_path),
                "--phase", "final", "--synthetic-actual", str(actual)]
        for name, file in files.items():
            argv.extend(["--synthetic-" + name, str(file)])
        argv.extend(["--trusted-synthetic-approval-sha256", "a" * 64,
                     "--trusted-synthetic-grant-sha256", "b" * 64])
        with patch.object(sys, "argv", argv), \
             patch.object(gate, "check_manifest", return_value={"fixture": True}) as check, \
             contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(gate.main(), 0)
        args, kwargs = check.call_args
        self.assertEqual(args, (self.manifest, "final"))
        for name, file in files.items():
            self.assertEqual(kwargs[name + "_bytes"], file.read_bytes())
        self.assertEqual(kwargs["trusted_approval_sha256"], "a" * 64)
        self.assertEqual(kwargs["trusted_grant_sha256"], "b" * 64)
        self.assertEqual(kwargs["actual"], {"fixture": True})

    def test_cli_never_defaults_or_derives_missing_trust_anchors(self):
        argv = ["validate_research_parity.py", "--manifest", str(self.manifest_path)]
        with patch.object(sys, "argv", argv), \
             patch.object(gate, "check_manifest", return_value={"fixture": True}) as check, \
             contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(gate.main(), 0)
        self.assertIsNone(check.call_args.kwargs["trusted_approval_sha256"])
        self.assertIsNone(check.call_args.kwargs["trusted_grant_sha256"])
        self.assertIsNone(check.call_args.kwargs["approval_bytes"])
        self.assertIsNone(check.call_args.kwargs["grant_bytes"])

    def test_adjacent_helper_loads_source_fresh_without_numerical_imports(self):
        first = gate._synthetic_policy()
        second = gate._synthetic_policy()
        self.assertIsInstance(first, types.ModuleType)
        self.assertIsNot(first, second)
        self.assertEqual(Path(first.__file__), Path(gate.__file__).with_name("synthetic_diagnostic_policy.py"))
        no_torch_probe.assert_isolated_no_torch(self, __file__)

    def test_existing_no_fit_path_does_not_load_synthetic_helper(self):
        manifest = {"schema_version": "tac-osm.parity.v1", "study_id": "NO_FIT_TEST_ONLY",
                    "study_type": "noncomparative_diagnostic", "no_new_model_fits": True,
                    "no_architecture_superiority_claim": True,
                    "scientific_purpose": "Source-only compatibility check"}
        with patch.object(gate, "_synthetic_policy", side_effect=AssertionError("unexpected helper")):
            self.assertEqual(gate.check_manifest(manifest)["status"], "PASS_DIAGNOSTIC_ONLY")


if __name__ == "__main__":
    unittest.main()
