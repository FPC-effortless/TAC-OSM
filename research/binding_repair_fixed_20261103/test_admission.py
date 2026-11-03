"""Source-only scalar admission fixtures. A test approval is never a run grant."""
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
import hashlib
from unittest.mock import patch
import types

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("repair_admission", HERE/"code/admission.py")
admission = importlib.util.module_from_spec(spec)
spec.loader.exec_module(admission)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class Scope(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)/"fixture"
        shutil.copytree(HERE, self.root, ignore=shutil.ignore_patterns("source", "fixtures", "runs", "__pycache__"))
        self.protocol = json.loads((self.root/"PROTOCOL.json").read_bytes())
        self.grant = json.loads((self.root/"REQUEST_TEMPLATE.json").read_bytes())

    def tearDown(self):
        self.temp.cleanup()

    def refresh(self):
        self.protocol["admission"]["files"]["OWNER_AUTHORIZATION.json"] = {"sha256": sha(self.root/"OWNER_AUTHORIZATION.json"), "bytes": (self.root/"OWNER_AUTHORIZATION.json").stat().st_size}
        (self.root/"PROTOCOL.json").write_text(json.dumps(self.protocol, indent=2)+"\n")
        self.grant["owner_authorization_sha256"] = sha(self.root/"OWNER_AUTHORIZATION.json")
        self.grant["protocol_sha256"] = sha(self.root/"PROTOCOL.json")

    def scalar_approval_fixture(self):
        # Temporary scalar fixture only. No model/optimizer entry point is called.
        p = self.root/"OWNER_AUTHORIZATION.json"
        data = json.loads(p.read_bytes())
        data.update(approved=True, user_response="Yes", basis="DIRECT_USER_RESPONSE_TO_THIS_EXACT_REPAIR_TRIAL")
        p.write_text(json.dumps(data)+"\n")
        self.protocol["training_authorized"] = True
        self.grant["authorized"] = True
        self.refresh()

    def result(self):
        self.assertNotIn("torch", sys.modules)
        return admission.inspect(self.root, self.protocol, self.grant)

    def test_current_real_proposal_remains_blocked(self):
        r = self.result()
        self.assertEqual(r["status"], "BLOCKED_REPAIR_TRIAL")
        self.assertFalse(r["old_one_off_exception_transferred"])

    def test_exact_scalar_approval_fixture_admits_no_execution(self):
        self.scalar_approval_fixture()
        r = self.result()
        self.assertEqual(r["status"], "PASS_FRESH_OWNER_SCOPED_REPAIR_DIAGNOSTIC")
        self.assertEqual(r["optimizer_updates_at_admission"], 0)

    def test_seed_change_cannot_self_rehash(self):
        self.scalar_approval_fixture()
        self.protocol["seed"] += 1
        self.refresh()
        self.assertEqual(self.result()["status"], "BLOCKED_REPAIR_TRIAL")

    def test_update_extension_cannot_self_rehash(self):
        self.scalar_approval_fixture()
        self.protocol["stages"]["fixed"]["updates_per_arm"] += 1
        self.refresh()
        self.assertEqual(self.result()["status"], "BLOCKED_REPAIR_TRIAL")

    def test_CPU_extension_cannot_self_rehash(self):
        self.scalar_approval_fixture()
        self.protocol["stages"]["fixed"]["caps"]["total_CPU"] += 1
        self.refresh()
        self.assertEqual(self.result()["status"], "BLOCKED_REPAIR_TRIAL")

    def test_old_approval_schema_cannot_transfer(self):
        self.scalar_approval_fixture()
        p = self.root/"OWNER_AUTHORIZATION.json"
        d = json.loads(p.read_bytes()); d["schema"] = "old_one_off"
        p.write_text(json.dumps(d)); self.refresh()
        self.assertEqual(self.result()["status"], "BLOCKED_REPAIR_TRIAL")

    def test_disabled_grant_blocks_even_scalar_approval(self):
        self.scalar_approval_fixture(); self.grant["authorized"] = False
        self.assertEqual(self.result()["status"], "BLOCKED_REPAIR_TRIAL")

    def test_retry_cannot_be_enabled(self):
        self.scalar_approval_fixture(); self.grant["retry_authorized"] = True
        self.assertEqual(self.result()["status"], "BLOCKED_REPAIR_TRIAL")

    def test_code_mutation_is_rejected(self):
        self.scalar_approval_fixture()
        with (self.root/"code/runner.py").open("a") as handle:
            handle.write("\n# changed\n")
        self.assertEqual(self.result()["status"], "BLOCKED_REPAIR_TRIAL")

    def test_same_job_factory_routes_control_and_repair_separately(self):
        spec = importlib.util.spec_from_file_location("repair_trial_core", HERE/"code/core.py")
        core = importlib.util.module_from_spec(spec); spec.loader.exec_module(core)
        parent = types.ModuleType("collect018"); parent.__path__ = []
        old = types.ModuleType("collect018.common")
        old.build = lambda arm, seed: ("legacy", arm, seed)
        new = types.ModuleType("model_repair")
        new.build = lambda family, seed: ("repaired", family, seed)
        with patch.dict(sys.modules, {"collect018": parent, "collect018.common": old, "model_repair": new}):
            self.assertEqual(core.build_trial("context_legacy_first", 18001), ("legacy", "context_first", 18001))
            self.assertEqual(core.build_trial("context_repaired_first", 18001), ("repaired", "context", 18001))
        self.assertNotIn("torch", sys.modules)

    def test_each_arm_uses_its_own_reference_equations(self):
        spec = importlib.util.spec_from_file_location("repair_trial_core", HERE/"code/core.py")
        core = importlib.util.module_from_spec(spec); spec.loader.exec_module(core)
        parent = types.ModuleType("collect018"); parent.__path__ = []
        old = types.ModuleType("collect018.reference")
        old.whole_reference = lambda model, token: "legacy-reference"
        new = types.ModuleType("repaired_reference")
        new.whole_reference = lambda model, token: "repaired-reference"
        with patch.dict(sys.modules, {"collect018": parent, "collect018.reference": old, "repaired_reference": new}):
            self.assertEqual(core.trial_reference(types.SimpleNamespace(variant="collect018_context"), None), "legacy-reference")
            self.assertEqual(core.trial_reference(types.SimpleNamespace(variant="FLOCK013_CONTEXT_ZERO_GAIN_ADJOINT"), None), "repaired-reference")
        self.assertNotIn("torch", sys.modules)


if __name__ == "__main__":
    unittest.main()
