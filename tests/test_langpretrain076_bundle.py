"""Validate authentic FIT-only single-code-file Kaggle packaging without GPUs."""
import ast
import base64
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
import zipfile
import zlib

from scripts.langpretrain076_fit_only import fit_sources
from scripts.langpretrain076_kaggle_bundle import build, source_members, BundleError
from scripts.validate_research_parity import check_manifest, check_model_parameter_target, ParityError
from tac_osm.phone_research_ops import decide

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "research/mathematical-engineering/langpretrain076/FAIR_COMPARISON.json"
PROPOSAL = ROOT / "research/phone_queue/langpretrain076.json"


class OfflineBundleTests(unittest.TestCase):
    def test_publisher_fit_only_no_dev_bytes(self):
        fit = fit_sources(ROOT)
        self.assertEqual(len(fit),14)
        self.assertIn("Doc/howto/unicode.rst",fit)
        self.assertNotIn("Doc/tutorial/controlflow.rst",fit)
        self.assertTrue(all(len(v)>=129 for v in fit.values()))

    def test_full_frozen_parity_without_global_budget_relaxation(self):
        manifest=json.loads(MANIFEST.read_text())
        result=check_manifest(manifest,"preflight")
        self.assertEqual(result["status"],"PASS_PREFIT_SINGLE_MODEL")
        self.assertFalse(result["eligible_for_architecture_superiority"])
        self.assertEqual(result["research_execution"]["status"],"PASS_INTERNAL_RESEARCH_METADATA")
        self.assertEqual([x["trainable_parameters"] for x in manifest["models"]],[499623,499623])
        with self.assertRaises(ParityError):
            check_model_parameter_target(499623,499623)
        manifest["parameter_budget_contract"]="incorrect-global-exemption"
        with self.assertRaises(ParityError):
            check_manifest(manifest,"preflight")

    def test_verified_embedded_fit_not_external_dataset_bypass(self):
        prop=json.loads(PROPOSAL.read_text())
        result=decide(prop)
        self.assertEqual(result["state"],"READY_INTERNAL_RESEARCH_PLAN")
        self.assertEqual(result["gpu_devices_required"],2)
        self.assertFalse(result["approved_to_train"])
        prop["run_id"]="unrelated-training-run"
        with self.assertRaises(ValueError):
            decide(prop)

    def test_whole_kaggle_bundle_is_single_code_file_and_has_no_dev(self):
        members=source_members(ROOT)
        original=[p for p in members if p.startswith(
            "research/source-bundles/LANGUAGE-SOURCE023/")]
        self.assertEqual(len(original),14)
        self.assertEqual(len(members),23)
        self.assertFalse(any(p.endswith("Doc/tutorial/controlflow.rst") for p in members))
        with tempfile.TemporaryDirectory() as d:
            folder=Path(d)/"kernel"
            result=build(ROOT,folder,"tacosmlab")
            self.assertFalse(result["approved_to_train"])
            self.assertFalse(result["approved_to_spend_paid"])
            self.assertEqual(result["DEV_bytes_embedded"],0)
            self.assertEqual(result["TEST_bytes_embedded"],0)
            self.assertEqual(sorted(p.name for p in folder.iterdir()),
                             ["entrypoint.py","kernel-metadata.json","staging-receipt.json"])
            code=(folder/"entrypoint.py").read_text()
            compile(code,"single_kaggle_code_file.py","exec") # NEVER execute GPU harness
            tree=ast.parse(code)
            globals_={}
            for node in tree.body:
                if isinstance(node,ast.Assign) and len(node.targets)==1 and isinstance(node.targets[0],ast.Name):
                    if node.targets[0].id in ("PAYLOAD","PAYLOAD_SHA256","MEMBERS"):
                        globals_[node.targets[0].id]=ast.literal_eval(node.value)
            raw=zlib.decompress(base64.b85decode(globals_["PAYLOAD"].encode("ascii")))
            self.assertEqual(hashlib.sha256(raw).hexdigest(),globals_["PAYLOAD_SHA256"])
            with zipfile.ZipFile(io.BytesIO(raw)) as z:
                self.assertEqual(set(z.namelist()),set(globals_["MEMBERS"]))
                self.assertEqual(len(z.namelist()),24) # 23 original files + generated config
                for name in z.namelist():
                    self.assertEqual(hashlib.sha256(z.read(name)).hexdigest(),globals_["MEMBERS"][name])
            metadata=json.loads((folder/"kernel-metadata.json").read_text())
            self.assertEqual(metadata["code_file"],"entrypoint.py")
            self.assertEqual(metadata["enable_internet"],"false")
            self.assertEqual(metadata["is_private"],"true")
            self.assertEqual(metadata["machine_shape"],"NvidiaTeslaT4")
            self.assertEqual(metadata["dataset_sources"],[])

    def test_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as d:
            folder=Path(d)/"kernel"
            folder.mkdir()
            with self.assertRaises(BundleError):
                build(ROOT,folder,"tacosmlab")


if __name__ == "__main__":
    unittest.main()
