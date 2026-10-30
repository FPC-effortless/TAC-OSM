"""Offline construction fixtures; no model scoring or capability claims."""
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np


SOURCE = Path(__file__).resolve().parents[1] / "research/mathematical-engineering/langpersist096/data.py"
SPEC = importlib.util.spec_from_file_location("langpersist096_data", SOURCE)
data = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(data)


def fixture_corpus():
    fit = {"fit.py": b"known_fit = 0\nfiller = known_fit\n"}
    documents, rows = {}, []
    for publisher_index, publisher in enumerate(data.PUBLISHERS):
        blocks = []
        for group in ("positive", "negative"):
            for index in range(30):
                name = f"Novel_{publisher_index}_{group[:3]}_{index:02d}".encode()
                first = name + b" = 1\n" if group == "positive" else b"filler = 1\n"
                padding = 192 - len(first)
                prefix = first + b"#" + b"x" * (padding - 2) + b"\n"
                blocks.append(prefix + name + b" = 2\n#" + b"z" * 210 + b"\n")
        raw = b"".join(blocks)
        filename = f"fixture-{publisher_index}.bin"
        documents[filename] = raw
        rows.append({"repo": publisher, "file": filename, "path": "fixture.py",
                     "partition": "TRANSFER_DEV", "sha256": data.digest(raw),
                     "git_blob_sha1": data.git_blob(raw), "size": len(raw)})
    return fit, documents, rows


class SourceAuthenticationTests(unittest.TestCase):
    def test_exact_raw_sha_blob_and_size(self):
        raw = b"NovelName = 1\r\n"
        row = {"path": "example.py", "sha256": data.digest(raw), "size": len(raw),
               "git_blob_sha1": data.git_blob(raw)}
        self.assertTrue(data.authenticate_bytes(raw, row)["all_three_identities_prefrozen"])
        for changed in ({**row, "size": len(raw) + 1}, {**row, "git_blob_sha1": "0" * 40},
                        {**row, "sha256": "0" * 64}):
            with self.subTest(changed=changed):
                with self.assertRaises(ValueError):
                    data.authenticate_bytes(raw, changed)

    def test_license_records_do_not_invent_prefrozen_size_or_blob(self):
        raw = b"Original fixture license text\n"
        measured = data.authenticate_bytes(raw, {"sha256": data.digest(raw)}, license_only=True)
        self.assertFalse(measured["all_three_identities_prefrozen"])
        self.assertEqual(measured["git_blob_sha1"], data.git_blob(raw))

    def test_source_cache_reuses_authenticated_bytes_without_overwrite(self):
        raw = b"SourceName = 1\n"
        row = {"repo": "fixture/publisher", "commit": "1" * 40, "path": "source.py",
               "file": "00.bin", "sha256": data.digest(raw), "git_blob_sha1": data.git_blob(raw),
               "size": len(raw)}
        with tempfile.TemporaryDirectory() as folder:
            cache = Path(folder)
            with patch.object(data.urllib.request, "urlopen", return_value=io.BytesIO(raw)):
                first, receipt = data._cached(cache, row)
            self.assertFalse(receipt["cache_hit"])
            with patch.object(data.urllib.request, "urlopen", side_effect=AssertionError("network forbidden on cache hit")):
                second, receipt = data._cached(cache, row)
            self.assertEqual(first, second)
            self.assertTrue(receipt["cache_hit"])
            (cache / "00.bin").write_bytes(b"corrupt")
            with self.assertRaises(ValueError):
                data._cached(cache, row)
            self.assertEqual((cache / "00.bin").read_bytes(), b"corrupt")

    def test_unknown_test_partition_rejected_before_payload_access(self):
        receipt = {"study": "CODEVALUE003", "files": [{"partition": "TEST"}]}
        identity = data.digest(data.canonical_json(receipt))
        receipt["cohort_sha256"] = identity
        with patch.object(data, "SOURCE_COHORT_SHA256", identity):
            with self.assertRaisesRegex(ValueError, "protected/unknown"):
                data._validate_receipt(receipt)


class IdentifierByteOffsetTests(unittest.TestCase):
    def test_unicode_prefix_and_bom_use_original_byte_offsets(self):
        for raw in ("é = 1; FreshIdentifier = 2\n".encode(),
                    b"\xef\xbb\xbfFreshIdentifier = 2\r\n",
                    b"# coding: latin-1\n\xe9 = 1; FreshIdentifier = 2\n"):
            with self.subTest(raw=raw):
                spans = data.ascii_name_spans(raw)
                expected = raw.index(b"FreshIdentifier")
                self.assertIn((expected, expected + len(b"FreshIdentifier"), b"FreshIdentifier"), spans)
                for start, end, name in spans:
                    self.assertEqual(raw[start:end], name)

    def test_comments_strings_unicode_names_and_keywords_excluded(self):
        raw = '# FakeComment\n"FakeString"\nαUnicode = 1\nwhile True:\n    RealName = 2\n'.encode()
        names = {name for _, _, name in data.ascii_name_spans(raw)}
        self.assertEqual(names, {b"RealName"})


class FrozenCohortTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fit, cls.documents, cls.rows = fixture_corpus()
        cls.built = data.build_cohort(cls.fit, cls.documents, cls.rows)

    def test_exact_shapes_balanced_strata_and_nonoverlap(self):
        built = self.built
        self.assertEqual(built["status"], "COHORT_READY")
        expected = {"acquisition.npy": (300, 128), "queries.npy": (300, 64),
                    "targets.npy": (300, 4), "donor_indices.npy": (300,)}
        for filename, shape in expected.items():
            self.assertEqual(built["arrays"][filename].shape, shape)
            self.assertEqual(built["arrays"][filename].dtype, np.int64 if filename.startswith("donor") else np.uint8)
        for publisher in data.PUBLISHERS:
            for group in data.GROUPS:
                self.assertEqual(sum(r["publisher"] == publisher and r["group"] == group for r in built["records"]), 20)
        for filename in self.documents:
            intervals = sorted((r["acquisition_start"], r["target_end"]) for r in built["records"] if r["file"] == filename)
            self.assertTrue(all(right[0] >= left[1] for left, right in zip(intervals, intervals[1:])))

    def test_selected_data_matches_real_contiguous_fixture_bytes(self):
        inventory = data.fit_identifier_inventory(self.fit)
        for record in self.built["records"]:
            index, raw = record["index"], self.documents[record["file"]]
            position = record["target_offset"]
            a, q = raw[position - 192:position - 64], raw[position - 64:position]
            self.assertEqual(self.built["arrays"]["acquisition.npy"][index].tobytes(), a)
            self.assertEqual(self.built["arrays"]["queries.npy"][index].tobytes(), q)
            self.assertEqual(self.built["arrays"]["targets.npy"][index].tobytes(), raw[position:position + 4])
            if record["group"] == "uniform":
                self.assertIsNone(record["target_name"])
                continue
            name = record["target_name"].encode()
            self.assertNotIn(name, inventory)
            self.assertNotIn(name, q)
            self.assertEqual(raw[position:position + len(name)], name)
            self.assertTrue(4 <= len(name) <= 24)
            if record["group"] == "positive":
                self.assertIn(name, a)
                self.assertIn((record["historical_name_offset"], record["historical_name_offset"] + len(name), name),
                              data.ascii_name_spans(raw))
            else:
                self.assertNotIn(name, a)

    def test_donors_are_fixed_cyclic_other_publishers_without_target_filter(self):
        publishers = sorted(data.PUBLISHERS)
        for record, donor in zip(self.built["records"], self.built["arrays"]["donor_indices.npy"]):
            expected = publishers[(publishers.index(record["publisher"]) + 1) % 5]
            self.assertEqual(self.built["records"][int(donor)]["publisher"], expected)
            self.assertEqual(int(donor) % 60, record["index"] % 60)

    def test_selection_deterministic_under_input_mapping_and_row_order(self):
        repeated = data.build_cohort(dict(reversed(list(self.fit.items()))),
                                    dict(reversed(list(self.documents.items()))), list(reversed(self.rows)))
        self.assertEqual(repeated["records"], self.built["records"])
        self.assertEqual(repeated["selection_proof"], self.built["selection_proof"])
        for name in self.built["arrays"]:
            np.testing.assert_array_equal(repeated["arrays"][name], self.built["arrays"][name])

    def test_insufficient_positive_counts_do_not_relax_or_emit_arrays(self):
        documents = {row["file"]: b"known_fit = 0\n" * 1500 for row in self.rows}
        result = data.build_cohort(self.fit, documents, self.rows)
        self.assertEqual(result["status"], "INSUFFICIENT_COHORT")
        self.assertNotIn("arrays", result)
        for publisher in data.PUBLISHERS:
            self.assertEqual(result["selection_proof"]["candidate_counts"][publisher]["positive"]["selected_nonoverlapping"], 0)


if __name__ == "__main__":
    unittest.main()
