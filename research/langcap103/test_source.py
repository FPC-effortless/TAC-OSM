"""Constructed integrity falsifiers only: no model fits or capability evidence."""
import copy
import hashlib
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest
from source import canonical, digest, git_blob, next_byte_batch, select_records, source_group, split_for, verify_package


PLAN = json.loads(Path(__file__).with_name("SOURCE_PLAN.json").read_text())


def row(name, data):
    return {"path": name, "bytes": len(data), "sha256": digest(data), "git_blob": git_blob(data)}


class SourceIntegrityTests(unittest.TestCase):
    def test_same_module_group_never_splits_children(self):
        self.assertEqual(source_group("Lib/asyncio/tasks.py"), source_group("Lib/asyncio/futures.py"))
        self.assertEqual(split_for(source_group("Lib/asyncio/tasks.py"), PLAN),
                         split_for(source_group("Lib/asyncio/futures.py"), PLAN))

    def test_parent_escape_is_rejected(self):
        with self.assertRaises(ValueError):
            source_group("Lib/../../outside.py")

    def test_all_exact_duplicates_are_excluded_before_split(self):
        records = [row("Lib/a.py", b"duplicate"), row("Lib/b.py", b"duplicate"), row("Lib/c.py", b"unique")]
        accepted, excluded, _ = select_records(records, PLAN)
        self.assertEqual([r["path"] for r in accepted], ["Lib/c.py"])
        self.assertEqual(len(excluded), 2)

    def test_manifest_is_input_order_independent(self):
        records = [row("Lib/c.py", b"c"), row("Lib/b.py", b"b"), row("Lib/a.py", b"a")]
        self.assertEqual(select_records(records, PLAN), select_records(records[::-1], PLAN))

    def test_shift_targets_and_fit_only_document_boundaries(self):
        r = {"identity": {"files": [
            {**row("Lib/a.py", b"abcdef"), "split": "FIT"},
            {**row("Lib/b.py", b"uvwxyz"), "split": "FIT"},
            {**row("Lib/dev.py", b"secret"), "split": "DEV"}]}}
        x, y, ids = next_byte_batch(r, {"Lib/a.py": b"abcdef", "Lib/b.py": b"uvwxyz",
            "Lib/dev.py": b"secret"}, length=5, batch=2)
        self.assertEqual(x[0], list(b"abcde")); self.assertEqual(y[0], list(b"bcdef"))
        self.assertEqual(x[1], list(b"uvwxy")); self.assertEqual(y[1], list(b"vwxyz"))
        self.assertEqual([z["path"] for z in ids], ["Lib/a.py", "Lib/b.py"])

    def test_short_documents_cannot_be_spliced_into_sequences(self):
        r = {"identity": {"files": [{**row("Lib/a.py", b"ab"), "split": "FIT"}]}}
        with self.assertRaises(ValueError):
            next_byte_batch(r, {"Lib/a.py": b"ab"}, length=2, batch=1)

    def package(self, folder, *, extra=False):
        b = b"authentic fixture bytes"; r = {**row("Lib/a.py", b), "group": "a", "split": "FIT"}
        tar_path = folder / "corpus.tar"
        with tarfile.open(tar_path, "w") as tar:
            entry = tarfile.TarInfo("Lib/a.py"); entry.size = len(b); tar.addfile(entry, io.BytesIO(b))
            if extra:
                entry = tarfile.TarInfo("../escape"); entry.size = 1; tar.addfile(entry, io.BytesIO(b"x"))
        identity = {"files": [r], "licenses": []}
        receipt = {"schema": "langcap103.publisher-corpus.v1", "identity": identity,
            "source_sha256": digest(canonical(identity)), "archive_sha256": digest(tar_path.read_bytes())}
        path = folder / "receipt.json"; path.write_text(json.dumps(receipt))
        return path, tar_path

    def test_complete_package_roundtrip(self):
        with tempfile.TemporaryDirectory() as d:
            r, t = self.package(Path(d)); _, files = verify_package(r, t)
            self.assertEqual(files["Lib/a.py"], b"authentic fixture bytes")

    def test_corrupted_archive_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            r, t = self.package(Path(d)); t.write_bytes(t.read_bytes() + b"corruption")
            with self.assertRaisesRegex(ValueError, "archive checksum"):
                verify_package(r, t)

    def test_changed_identity_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            r, t = self.package(Path(d)); doc = json.loads(r.read_text())
            doc["identity"]["files"][0]["split"] = "DEV"; r.write_text(json.dumps(doc))
            with self.assertRaisesRegex(ValueError, "identity checksum"):
                verify_package(r, t)

    def test_unsafe_extra_archive_member_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            r, t = self.package(Path(d), extra=True)
            with self.assertRaisesRegex(ValueError, "unsafe"):
                verify_package(r, t)


if __name__ == "__main__":
    unittest.main()
