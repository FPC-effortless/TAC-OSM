"""GRAPH013 constructed software contracts only, not real-model evaluations."""
import io
import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from graph013_500k_archive_replay import _sha, _safe_checkpoint, replay
from tac_wave_ced_500k import WaveCED500K

class ReplayContracts(unittest.TestCase):
    def test_digest_is_byte_sensitive(self):
        self.assertEqual(len(_sha(b"source")), 64)
        self.assertNotEqual(_sha(b"source"), _sha(b"SOURCE"))

    def test_restricted_numpy_metadata_load(self):
        fixture = {"model":{"w":torch.tensor([1.0])},
                   "numpy_rng":np.random.get_state(),
                   "dev":{"nll":np.float64(2.0)}}
        buf=io.BytesIO()
        torch.save(fixture,buf)
        got=_safe_checkpoint(buf.getvalue())
        self.assertEqual(got["dev"]["nll"],2.0)
        self.assertTrue(torch.equal(got["model"]["w"],fixture["model"]["w"]))

    def test_absent_archive_fails_closed(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaisesRegex(ValueError,"archive is absent"):
                replay(Path(d)/"absent.zip",Path(d))

    def test_original_model_has_499290_learned_parameters(self):
        model=WaveCED500K()
        self.assertEqual(model.parameter_count(),499290)
        self.assertEqual(sum(p.numel() for p in model.parameters() if p.requires_grad),499290)

if __name__=="__main__":
    unittest.main()
