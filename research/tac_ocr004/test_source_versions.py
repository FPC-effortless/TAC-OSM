"""Pure boundary regressions; no historical implementation is imported."""
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[2]
SPEC=importlib.util.spec_from_file_location('source_versions_contract',Path(__file__).with_name('verify_source_versions.py'))
contract=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(contract)

class SourceVersions(unittest.TestCase):
    def no_remote(self,url):self.fail('unexpected remote read')

    def copy_boundary(self,base):
        path='research/tac_ocr004/SOURCE_VERSIONS.json'
        data=json.loads((ROOT/path).read_text())
        for name in [path,*data['unchanged_local_legacy_files'],*data['mandatory_current_contract']]:
            target=base/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,target)
        return data

    def test_missing_manifest_rejects_before_network(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):contract.verify(Path(tmp),self.no_remote)

    def test_moving_historical_ref_rejects(self):
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp);self.copy_boundary(base)
            path=base/'research/tac_ocr004/SOURCE_VERSIONS.json';data=json.loads(path.read_text());data['historical_source_commit']='master';path.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError,'manifest drift'):contract.verify(base,self.no_remote)

    def test_changed_original_validator_rejects_before_execution(self):
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp);self.copy_boundary(base)
            path=base/'research/tac_ocr001/verify_math_pins.py';path.write_bytes(path.read_bytes()+b'\nraise RuntimeError("must not execute")\n')
            with self.assertRaisesRegex(ValueError,'local file drift'):contract.verify(base,self.no_remote)

    def test_current_contract_failure_is_mandatory(self):
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp);self.copy_boundary(base)
            # The real current checker reads local registry before its peer.
            for name in ['research/pnds-variant-math-links.json',*[f'research/tac_ocr004/{f}/MATH_SOURCE_MAP.json' for f in ['tac','transformer','gru']]]:
                target=base/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,target)
            def reject_current(url):raise ValueError('current gate sentinel')
            with self.assertRaisesRegex(ValueError,'current gate sentinel'):contract.verify(base,reject_current)

    def test_original_default_validator_bytes_unchanged(self):
        data=json.loads((ROOT/'research/tac_ocr004/SOURCE_VERSIONS.json').read_text())
        for path,expected in data['unchanged_local_legacy_files'].items():self.assertEqual(hashlib.sha256((ROOT/path).read_bytes()).hexdigest(),expected)

if __name__=='__main__':unittest.main()
