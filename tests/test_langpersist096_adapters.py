"""Constructed state/source checks only; no trained quality evidence."""
from dataclasses import replace
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
try:
    import torch
except ImportError:
    torch = None

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "research/mathematical-engineering/langpersist096"


@unittest.skipIf(torch is None, "optional PyTorch unavailable")
class AdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(STUDY))
        spec = importlib.util.spec_from_file_location("langpersist096_adapters", STUDY / "adapters.py")
        cls.a = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = cls.a
        spec.loader.exec_module(cls.a)
        torch.set_num_threads(1)

    def model(self, arm):
        torch.manual_seed(96096)
        model = self.a.TRAIN.make(arm).float().eval()
        model._checkpoint_sha256_096 = "constructed_fixture_not_training_evidence"
        return model

    def test_whole_source_prefix_equals_acquisition_query_and_chunked_steps(self):
        x = torch.randint(0, 256, (2, 192))
        for arm in ("tac", "gru", "ssm"):
            with self.subTest(arm=arm):
                model = self.model(arm); adapter = self.a.FrozenAdapter096(model, arm)
                with torch.inference_mode():
                    expected = model(x)
                acquired = self.a.acquire(model, arm, x[:, :128], ("one", "two"))
                actual, state = adapter.observe(acquired, x[:, 128:])
                torch.testing.assert_close(actual, expected[:, -1], atol=3e-5, rtol=1e-4)
                next_logits, _ = adapter.step(state, torch.tensor([4, 9]))
                with torch.inference_mode():
                    joined = model(torch.cat((x, torch.tensor([[4], [9]])), 1))[:, -1]
                torch.testing.assert_close(next_logits, joined, atol=3e-5, rtol=1e-4)
                self.assertEqual(state.clock, 192)
                self.assertEqual(set(state.tensors), set(adapter.shapes))
                self.assertTrue(all(v.shape[0] == 2 for v in state.tensors.values()))

    def test_reset_donor_corrupt_cover_every_tensor_preserve_clock_and_owners(self):
        for arm in ("tac", "gru", "ssm"):
            adapter = self.a.FrozenAdapter096(self.model(arm), arm)
            state = adapter.observe(adapter.start(("one", "two")), torch.randint(0, 256, (2, 128)))[1]
            original = {k: v.clone() for k, v in state.tensors.items()}
            for condition in self.a.CONDITIONS:
                result = self.a.intervene(state, condition, np.array([1, 0], np.int64))
                self.assertEqual(result.clock, 128); self.assertEqual(result.owners, state.owners)
                adapter.validate(result)
                for name, value in result.tensors.items():
                    if condition == "reset":
                        self.assertFalse(bool(value.any()), name)
                    elif condition == "donor":
                        torch.testing.assert_close(value, state.tensors[name].flip(0), atol=0, rtol=0)
                    elif condition == "corrupt":
                        torch.testing.assert_close(value.flatten(1).norm(dim=1), state.tensors[name].flatten(1).norm(dim=1), atol=1e-5, rtol=1e-5)
                    torch.testing.assert_close(state.tensors[name], original[name], atol=0, rtol=0)
            with self.assertRaises(ValueError):
                self.a.intervene(state, "donor", np.array([0, 1]))
            with self.assertRaises(ValueError):
                adapter.validate(replace(state, checkpoint_sha256="different"))

    def test_state_snapshot_accounting_and_strict_checkpoint_hash(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            for arm in ("tac", "gru", "ssm"):
                model = self.model(arm); cp = folder / (arm+".pt")
                torch.save(dict(model=model.state_dict(), fixture="UNTRAINED_ENGINEERING_ONLY"), cp)
                loaded = self.a._construct_model(arm, {"model": model.state_dict()}, "constructed_fixture")
                adapter = self.a.FrozenAdapter096(loaded, arm)
                state = adapter.observe(adapter.start(("a", "b")), torch.randint(0, 256, (2, 128)))[1]
                path = folder / (arm+"-state.pt")
                meta = self.a.save_state(path, state, {"acquisition_seconds": .1}, "0"*64)
                restored, read = self.a.load_state(path, adapter)
                self.assertEqual(meta, read)
                self.assertEqual(meta["cost"]["state_payload_bytes"], state.tensor_bytes())
                for name in state.tensors:
                    torch.testing.assert_close(state.tensors[name], restored.tensors[name], atol=0, rtol=0)
                with self.assertRaises(ValueError):
                    self.a.make_model(arm, cp, "0"*64)
                with self.assertRaises(ValueError):
                    self.a.make_model(arm, cp, self.a.sha256(cp))

    def test_query_subprocess_does_not_need_acquisition_array(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder); model = self.model("gru")
            cp = folder / "model.pt"
            torch.save(dict(model=model.state_dict(), fixture="UNTRAINED_ENGINEERING_ONLY"), cp)
            acquisition = folder / "acquisition.npy"; queries = folder / "queries.npy"
            donors = folder / "donors.npy"; state = folder / "state.pt"
            np.save(acquisition, np.arange(256, dtype=np.uint8).reshape(2, 128))
            np.save(queries, np.zeros((2, 64), dtype=np.uint8))
            np.save(donors, np.array([1, 0], dtype=np.int64))
            bootstrap = ("import sys;sys.path.insert(0," + repr(str(STUDY)) + ");import worker,adapters,torch;"
                         "worker.make_model=lambda arm,path,digest:adapters._construct_model(arm,torch.load(path,weights_only=True),digest);worker.main()")
            common = [sys.executable, "-c", bootstrap, "--arm", "gru", "--checkpoint", str(cp), "--sha256", self.a.sha256(cp), "--state", str(state)]
            subprocess.run(common+["--mode", "acquire", "--acquisition", str(acquisition), "--output", str(folder / "acquire.json")], check=True, capture_output=True)
            acquisition.unlink()  # query process cannot reopen the acquisition input
            output = folder / "predictions.npz"
            subprocess.run(common+["--mode", "query", "--queries", str(queries), "--donors", str(donors), "--output", str(output)], check=True, capture_output=True)
            with np.load(output, allow_pickle=False) as result:
                self.assertEqual(result["log_probs"].shape, (4, 2, 256))
                self.assertEqual(result["greedy"].shape, (4, 2, 4))
                metadata = json.loads(str(result["metadata"]))
                self.assertEqual(metadata["conditions"], ["carry", "reset", "donor", "corrupt"])
                self.assertEqual(metadata["optimizer_steps"], 0)
                self.assertGreater(metadata["cost"]["acquisition_seconds"], 0)
                self.assertEqual(set(metadata["per_condition_costs"]), set(self.a.CONDITIONS))


if __name__ == "__main__":
    unittest.main()
