"""Constructed ZIP integrity fixtures only; never download real artifacts."""

import hashlib
import io
import json
from pathlib import Path
import stat
import tempfile
import unittest
from unittest.mock import patch
import warnings
import zipfile

from research.langdepth106.collect_artifacts import (
    ArtifactError, EXPECTED_HEAD_SHA, EXPECTED_RUN_ID, RECEIPT_NAME,
    collect_artifacts,
)


def fixture_zip(members):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            for name, payload in members:
                archive.writestr(name, payload)
    return buffer.getvalue()


def fixture_manifest(data, name="fit4-seed1"):
    return {"artifacts": [{
        "id": 123, "name": name, "size_in_bytes": len(data),
        "digest": "sha256:" + hashlib.sha256(data).hexdigest(),
        "workflow_run": {"id": EXPECTED_RUN_ID, "head_sha": EXPECTED_HEAD_SHA},
        "download_url": "https://example.invalid/private?token=DO_NOT_PUBLISH",
    }]}


def stage_original(path, data, name="fit4-seed1"):
    folder = path / "original_artifacts"
    folder.mkdir(parents=True)
    (folder / (name + ".zip")).write_bytes(data)


class ArtifactCollectorTests(unittest.TestCase):
    def test_reuses_safe_original_and_evidence_excludes_urls(self):
        data = fixture_zip([("model/state.bin", b"constructed bytes"), ("result.json", b'{"fixture":true}')])
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            stage_original(output, data)
            manifest = fixture_manifest(data)
            with patch("research.langdepth106.collect_artifacts._download", side_effect=AssertionError("network must not run")):
                receipt = collect_artifacts(manifest, output, verify_existing=True)
                before = (output / RECEIPT_NAME).read_bytes()
                again = collect_artifacts(manifest, output, verify_existing=True)
            self.assertEqual(receipt, again)
            self.assertEqual(before, (output / RECEIPT_NAME).read_bytes())
            self.assertEqual((output / "original_artifacts/fit4-seed1.zip").read_bytes(), data)
            self.assertEqual((output / "extracted/fit4-seed1/model/state.bin").read_bytes(), b"constructed bytes")
            text = json.dumps(receipt)
            for forbidden in ("https://", "download_url", "DO_NOT_PUBLISH", "token="):
                self.assertNotIn(forbidden, text)
            files = receipt["artifacts"][0]["files"]
            self.assertEqual(files[0]["sha256"], hashlib.sha256(b"constructed bytes").hexdigest())
            self.assertEqual(receipt["artifacts"][0]["total_uncompressed_bytes"], len(b"constructed bytes") + len(b'{"fixture":true}'))
            (output / "extracted/fit4-seed1/unrelated").write_bytes(b"do not silently include")
            with self.assertRaises(ArtifactError):
                collect_artifacts(manifest, output, verify_existing=True)

    def test_digest_size_and_cohort_mismatch_fail_closed(self):
        data = fixture_zip([("file", b"safe")])
        mutations = [
            lambda a: a.update(digest="sha256:" + "0" * 64),
            lambda a: a.update(size_in_bytes=len(data) + 1),
            lambda a: a["workflow_run"].update(id=EXPECTED_RUN_ID + 1),
            lambda a: a["workflow_run"].update(head_sha="0" * 40),
            lambda a: a.update(name="../unsafe"),
            lambda a: a.update(download_url="http://example.invalid/DO_NOT_PUBLISH"),
        ]
        for index, mutate in enumerate(mutations):
            with self.subTest(mutation=index), tempfile.TemporaryDirectory() as tmp:
                output = Path(tmp)
                stage_original(output, data)
                manifest = fixture_manifest(data)
                mutate(manifest["artifacts"][0])
                with self.assertRaises(ArtifactError) as caught:
                    collect_artifacts(manifest, output, verify_existing=True)
                self.assertNotIn("DO_NOT_PUBLISH", str(caught.exception))
                self.assertFalse((output / RECEIPT_NAME).exists())
                self.assertEqual((output / "original_artifacts/fit4-seed1.zip").read_bytes(), data)

    def test_traversal_duplicate_symlink_and_parent_collision_rejected(self):
        symlink = zipfile.ZipInfo("link")
        symlink.create_system = 3
        symlink.external_attr = (stat.S_IFLNK | 0o777) << 16
        cases = [
            [("../outside", b"bad")], [("/absolute", b"bad")],
            [("dir\\windows", b"bad")], [("C:/drive", b"bad")],
            [("same", b"first"), ("same", b"second")],
            [(symlink, b"../outside")], [("parent", b"file"), ("parent/child", b"bad")],
        ]
        for index, members in enumerate(cases):
            with self.subTest(archive=index), tempfile.TemporaryDirectory() as tmp:
                output = Path(tmp)
                data = fixture_zip(members)
                stage_original(output, data)
                with self.assertRaises(ArtifactError):
                    collect_artifacts(fixture_manifest(data), output, verify_existing=True)
                self.assertFalse((output / RECEIPT_NAME).exists())
                self.assertFalse((output / "extracted/fit4-seed1/parent").exists())

    def test_bounded_expansion_and_existing_member_refuse_clobber(self):
        data = fixture_zip([("file", b"123456")])
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            stage_original(output, data)
            with patch("research.langdepth106.collect_artifacts.MAX_FILE_BYTES", 5):
                with self.assertRaises(ArtifactError):
                    collect_artifacts(fixture_manifest(data), output, verify_existing=True)
            collect_artifacts(fixture_manifest(data), output, verify_existing=True)
            member = output / "extracted/fit4-seed1/file"
            member.write_bytes(b"user modification")
            with self.assertRaises(ArtifactError):
                collect_artifacts(fixture_manifest(data), output, verify_existing=True)
            self.assertEqual(member.read_bytes(), b"user modification")
        data = fixture_zip([("one", b"1234"), ("two", b"5678")])
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            stage_original(output, data)
            with patch("research.langdepth106.collect_artifacts.MAX_ARTIFACT_BYTES", 7):
                with self.assertRaises(ArtifactError):
                    collect_artifacts(fixture_manifest(data), output, verify_existing=True)

    def test_offline_missing_original_and_extraction_symlinks_rejected(self):
        data = fixture_zip([("directory/file", b"safe")])
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            with patch("research.langdepth106.collect_artifacts._download", side_effect=AssertionError("network must not run")):
                with self.assertRaises(ArtifactError):
                    collect_artifacts(fixture_manifest(data), output, verify_existing=True)
            (output / "original_artifacts/fit4-seed1.zip").write_bytes(data)
            outside = output / "outside"
            outside.mkdir()
            target = output / "extracted/fit4-seed1"
            target.mkdir()
            (target / "directory").symlink_to(outside, target_is_directory=True)
            with self.assertRaises(ArtifactError):
                collect_artifacts(fixture_manifest(data), output, verify_existing=True)
            self.assertFalse((outside / "file").exists())


if __name__ == "__main__":
    unittest.main()
