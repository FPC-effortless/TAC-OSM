"""GRAPH011 numerical contracts against authentic source-code bytes.

This is a STORAGE/REPLAY correctness test, NOT a real language quality
benchmark. Fixture is the actual checked-in GRAPH009 Python implementation.
It is neither synthetic training data nor an independent model TEST split.
"""
import hashlib
import unittest
from pathlib import Path

import torch

from source_checkpoint_replay_v7 import VerifiedSparseReplay
from source_calibrated_tac_v5 import SourceCalibratedTAC
from tac_wave_ced_500k_v4 import SlotTAC500K


class SparseReplayContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.text = Path(__file__).with_name("bayesian_cache_scan_v5.py").read_bytes()
        if any(v >= 128 for v in cls.text):
            raise AssertionError("source fixture must be transparently ASCII model input")
        cls.text_sha = hashlib.sha256(cls.text).hexdigest()

    def setUp(self):
        torch.manual_seed(711)
        self.m=SourceCalibratedTAC(
            SlotTAC500K().eval(),torch.ones(128)/128,capacity=32).eval()
        self.m.requires_grad_(False)
        self.source=self.text[:65]
        self.digest=hashlib.sha256(self.source).hexdigest()

    def index(self,block):
        return VerifiedSparseReplay(
            self.m,self.source,owner="repo-source-owner",block=block,
            source_sha256=self.digest).prepare()

    def test_real_source_sparse_replay_same_as_full_and_dense(self):
        # Float32 PyTorch distributed/chunked execution is checked with an
        # explicit tolerance, NOT falsely advertised as IEEE bit equality.
        with torch.inference_mode():
            x=torch.tensor(list(self.source),dtype=torch.long)[None,:]
            reference=self.m(x,source_ids=("repo-source-owner",))
        dense=self.index(1)
        sparse=self.index(16)
        for t in (1,4,17,32,33,55,65):
            a,_,ar=dense.predict_at(t,owner="repo-source-owner",expected_sha256=self.digest)
            b,_,br=sparse.predict_at(t,owner="repo-source-owner",expected_sha256=self.digest)
            self.assertTrue(torch.allclose(a,b,atol=3e-4,rtol=3e-4))
            self.assertTrue(torch.allclose(a,reference[:,t-1,:],atol=3e-4,rtol=3e-4))
            self.assertLessEqual(br.replayed_tokens,16)
            self.assertEqual(ar.replayed_tokens,1)
            self.assertEqual(br.source_bytes,len(self.source))
            self.assertEqual(br.owner,"repo-source-owner")
            self.assertEqual(br.target_ordinal,t)
            self.assertEqual(br.model_sha256,dense.weights_sha256)
            self.assertEqual(br.checkpoint_bytes,sparse.resident_checkpoint_bytes())
        self.assertLess(sparse.resident_checkpoint_bytes(),dense.resident_checkpoint_bytes())
        self.assertEqual(len(dense.checkpoints),len(self.source))
        self.assertEqual(len(sparse.checkpoints),5)
        self.assertGreater(sparse.model_weight_bytes,0)

    def test_fail_closed_wrong_identity_source_and_model(self):
        index=self.index(8)
        with self.assertRaises(PermissionError):
            index.predict_at(12,owner="different-tenant",expected_sha256=self.digest)
        with self.assertRaisesRegex(ValueError,"source"):
            index.predict_at(12,owner="repo-source-owner",expected_sha256="00"*32)
        with self.assertRaisesRegex(ValueError,"target"):
            index.predict_at(0,owner="repo-source-owner",expected_sha256=self.digest)
        with self.assertRaisesRegex(ValueError,"target"):
            index.predict_at(len(self.source)+1,owner="repo-source-owner",expected_sha256=self.digest)
        index.source=b"X"+index.source[1:]
        with self.assertRaisesRegex(ValueError,"source"):
            index.predict_at(12,owner="repo-source-owner",expected_sha256=self.digest)
        index.source=self.source
        with torch.no_grad():
            self.m.fit_prior[0]+=0.0001
        with self.assertRaisesRegex(ValueError,"weight"):
            index.predict_at(12,owner="repo-source-owner",expected_sha256=self.digest)

    def test_source_authorization_and_frozen_model_preconditions(self):
        with self.assertRaisesRegex(ValueError,"hash"):
            VerifiedSparseReplay(self.m,self.source,owner="O",block=8,source_sha256="f"*64)
        with self.assertRaisesRegex(ValueError,"freeze"):
            VerifiedSparseReplay(SlotTAC500K().train(),self.source,
                owner="O",block=8,source_sha256=self.digest)
        with self.assertRaises(ValueError):
            VerifiedSparseReplay(self.m,self.source,owner="O",block=0,source_sha256=self.digest)
        with self.assertRaises(ValueError):
            VerifiedSparseReplay(self.m,b"\xff",owner="O",block=8,
                source_sha256=hashlib.sha256(b"\xff").hexdigest())


if __name__=="__main__":
    unittest.main()
