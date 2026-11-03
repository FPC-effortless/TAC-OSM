"""Pure OCR004 contract regressions: no model, network or peer activation."""
import copy
import importlib.util
from pathlib import Path
import unittest

spec=importlib.util.spec_from_file_location('ocr004_contract',Path(__file__).with_name('verify_immutable_peer.py'))
contract=importlib.util.module_from_spec(spec)
spec.loader.exec_module(contract)

class ContractInputs(unittest.TestCase):
    def test_frozen_maps_and_rows(self):
        for family in contract.FAMILIES:
            contract.validate_map(copy.deepcopy(contract.EXPECTED_MAPS[family]),family)
        rows=[contract.expected_row(f) for f in contract.FAMILIES]
        self.assertEqual(contract.select_rows({'prospective_candidates':rows}),rows)

    def test_map_drift_rejects(self):
        changes=[{'lean_compiled':True},{'tac_source_paths':[]},
                 {'mathematical_files_sha256':{}},{'lean_module':'../other.lean'},
                 {'open_obligations':[]},{'unexpected_authority':True}]
        for change in changes:
            with self.subTest(change=change):
                data=copy.deepcopy(contract.EXPECTED_MAPS['tac']);data.update(change)
                with self.assertRaises(ValueError):contract.validate_map(data,'tac')

    def test_missing_duplicate_unknown_rows_reject(self):
        rows=[contract.expected_row(f) for f in contract.FAMILIES]
        for bad in [rows[:-1],rows+[rows[0]],rows+[dict(rows[0],id='TAC-OCR-004-UNKNOWN')]]:
            with self.assertRaises(ValueError):contract.select_rows({'prospective_candidates':bad})

    def test_pin_guards_precede_io(self):
        base={'schema':'pnds.tac_ocr004.immutable_peer.v1',
              'peer_repository':contract.REPOS['pnds'],'metadata_commit':'main',
              'map_sha256':{f:'0'*64 for f in contract.FAMILIES}}
        def no_network(url):self.fail('unexpected network')
        with self.assertRaisesRegex(ValueError,'immutable commit'):
            contract.verify_immutable_peer('/nonexistent','tac',base,no_network)
        base['metadata_commit']='0'*40;base['peer_repository']='other/repository'
        with self.assertRaisesRegex(ValueError,'repository'):
            contract.verify_immutable_peer('/nonexistent','tac',base,no_network)

if __name__=='__main__':unittest.main()
