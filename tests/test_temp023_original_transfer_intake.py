"""TEMP023 fixture-only admission contracts, not actual neural replay."""
import hashlib
import io
import unittest
import warnings
from zipfile import ZipFile
from scripts.audit_temp023_original_transfer import (
    check_manifest,check_member_names,check_rights,
    ARCHIVE_BYTES,CHUNK_BYTES,PROTOCOL_SHA
)
class IntakeContracts(unittest.TestCase):
    def test_altered_original_archive_refused(self):
        m={"schema":"tac-osm.original-archive-shards.v1",
           "archive_sha256":"tampered","archive_bytes":ARCHIVE_BYTES,
           "chunk_bytes":CHUNK_BYTES,"chunks":[]}
        with self.assertRaises(ValueError):check_manifest(m)

    def test_zip_path_traversal_refused(self):
        raw=io.BytesIO()
        with ZipFile(raw,"w") as z:z.writestr("../unsafe","x")
        raw.seek(0)
        with ZipFile(raw) as z:
            with self.assertRaises(ValueError):check_member_names(z.infolist())

    def test_duplicate_names_refused(self):
        raw=io.BytesIO()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with ZipFile(raw,"w") as z:
                z.writestr("a","1")
                z.writestr("a","2")
        raw.seek(0)
        with ZipFile(raw) as z:
            with self.assertRaises(ValueError):check_member_names(z.infolist())

    def test_rights_cannot_be_self_approved(self):
        license_bytes=b"License: OPL-1+\n"
        notice=b"The Open Publication License applies."
        rights={"publisher_url":"https://github.com/vim/vim",
                "source_manifest_sha256":PROTOCOL_SHA,
                "license_sha256":hashlib.sha256(license_bytes).hexdigest(),
                "license":"Open Publication License",
                "redistribution_reviewed":False}
        self.assertEqual(check_rights(rights,license_bytes,notice),
                         "PUBLICATION_REVIEW_REQUIRED")
        rights["redistribution_reviewed"]=True
        with self.assertRaises(ValueError):check_rights(rights,license_bytes,notice)
if __name__=="__main__":unittest.main()
