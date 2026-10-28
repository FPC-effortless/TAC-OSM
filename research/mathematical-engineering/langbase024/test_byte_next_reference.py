"""LANGBASE024 constructed safety/causality regressions, not model skill."""
import hashlib
from pathlib import Path
import tempfile
import unittest
from byte_next_reference import law,train_count_model,score_document,read_sha_source

class ByteReferenceContracts(unittest.TestCase):
    def test_source_causal_suffix_independent_of_future(self):
        model=train_count_model({"fitA":b"abcabcabc", "fitB":b"xyzz"})
        for t in range(1,5):
            a=b"abcaXxyz"
            b=b"abcaY123"
            suffix_a=a[max(0,t-2):t+1]
            suffix_b=b[max(0,t-2):t+1]
            if t<=3:
                self.assertEqual(suffix_a,suffix_b)
                self.assertEqual(law(model,suffix_a,3),law(model,suffix_b,3))

    def test_probability_law_full_256_and_strictly_positive(self):
        model=train_count_model({"one":b"a"*100+b"\xff\xb0"})
        for history in (b"",b"a",b"aaa",b"xyz",b"\xff"):
            for order in (0,3):
                probs=law(model,history,order)
                self.assertEqual(len(probs),256)
                self.assertTrue(all(p>0 for p in probs))
                self.assertAlmostEqual(sum(probs),1.,places=10)

    def test_byte_supplement_fit_only_full_rare_target(self):
        model=train_count_model({"source":b"\x00\xff\xfe\x00"})
        self.assertEqual(model["targets"],3)
        self.assertEqual(model["nonascii_fit_targets"],2)
        score=score_document(model,b"\xff\xfe",3)
        self.assertEqual(score["targets"],1)
        self.assertEqual(score["nonascii_targets"],1)
        self.assertIsNotNone(score["nonascii_nll_nats_per_byte"])

    def test_document_crossing_forbidden_in_FIT_counts(self):
        separate=train_count_model({"a":b"abcdef","b":b"ghijkl"})
        merged=train_count_model({"a":b"abcdefghijkl"})
        self.assertEqual(separate["targets"],10)
        self.assertEqual(merged["targets"],11)
        self.assertNotIn(b"f",separate["counts"][1])
        self.assertIn(b"f",merged["counts"][1])

    def test_original_byte_sha_and_git_blob_fail_closed(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);p=root/"research/source-bundles/LANGUAGE-SOURCE023"
            p.mkdir(parents=True)
            (p/"test.rst").write_bytes(b"hello")
            data=b"hello"
            blob=hashlib.sha1(b"blob 5\x00"+data).hexdigest()
            receipt={"bytes":5,"sha256":hashlib.sha256(data).hexdigest(),"git_blob":blob}
            self.assertEqual(read_sha_source(root,"test.rst",receipt),data)
            (p/"test.rst").write_bytes(b"Hello")
            with self.assertRaises(ValueError):read_sha_source(root,"test.rst",receipt)
            with self.assertRaises(ValueError):read_sha_source(root,"../test.rst",receipt)

    def test_registered_hyperparameter_guard(self):
        model=train_count_model({"fit":b"byte-complete"})
        with self.assertRaises(ValueError):law(model,b"x",2)

if __name__=="__main__":unittest.main()
