"""Unit falsifiers for the scalar approval gate. No neural/model work."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'code'))
import admission


class Tests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.root=Path(self.temp.name)/'case'
        shutil.copytree(ROOT,self.root,ignore=shutil.ignore_patterns('source','fixtures','runs','__pycache__'))
        self.protocol=json.loads((ROOT/'PROTOCOL.json').read_bytes())

    def tearDown(self):self.temp.cleanup()

    def inspect(self,stage='fixed',regime='default'):
        return admission.inspect(self.root,self.protocol,stage,regime)

    def test_exact_scope_passes_with_old_rejections_retained(self):
        x=self.inspect()
        self.assertEqual(x['status'],'PASS_ONE_OFF_OWNER_SYNTHETIC_DIAGNOSTIC')
        self.assertFalse(x['eligible_for_architecture_superiority'])
        self.assertFalse(x['protected_TEST_authorized'])
        self.assertEqual(len(x['inherited_receipt']['rows']),3)
        self.assertTrue(all(not r['admitted'] for r in x['inherited_receipt']['rows']))

    def test_other_matrices_remain_blocked(self):
        for stage,regime in [('fixed','larger_epsilon'),('varying','default'),('varying','larger_epsilon')]:
            with self.subTest(stage=stage,regime=regime):
                self.assertEqual(self.inspect(stage,regime)['status'],'BLOCKED_EXISTING_POLICY')

    def test_seed_change_rejected(self):
        self.protocol['seed']+=1
        with self.assertRaises(ValueError):self.inspect()

    def test_update_extension_rejected(self):
        self.protocol['stages']['fixed']['updates_per_arm']=129
        with self.assertRaises(ValueError):self.inspect()

    def test_cpu_extension_rejected(self):
        self.protocol['stages']['fixed']['caps']['total_CPU']=361
        with self.assertRaises(ValueError):self.inspect()

    def test_wall_extension_rejected(self):
        self.protocol['stages']['fixed']['caps']['total_wall']=601
        with self.assertRaises(ValueError):self.inspect()

    def test_parameter_change_rejected(self):
        self.protocol['parameter_counts']['phase']=99972
        with self.assertRaises(ValueError):self.inspect()

    def test_optimizer_change_rejected(self):
        self.protocol['regimes']['default']['epsilon']=1e-6
        with self.assertRaises(ValueError):self.inspect()

    def test_data_change_rejected(self):
        self.protocol['fixture_files']['fixed_train_cases.json']['sha256']='0'*64
        with self.assertRaises(ValueError):self.inspect()

    def test_source_change_rejected(self):
        p=self.root/'code/runner.py';p.write_text(p.read_text()+'\n')
        with self.assertRaises(ValueError):self.inspect()

    def test_owner_approval_change_rejected(self):
        p=self.root/'OWNER_AUTHORIZATION.json';v=json.loads(p.read_text());v['user_response']='No';p.write_text(json.dumps(v))
        with self.assertRaises((ValueError,AssertionError)):self.inspect()

    def test_removed_approval_rejected(self):
        (self.root/'OWNER_AUTHORIZATION.json').unlink()
        with self.assertRaises((ValueError,FileNotFoundError)):self.inspect()

    def test_exception_forgery_rejected_even_if_manifest_is_rehashed(self):
        p=self.root/'APPROVED_OWNER_EXCEPTION.json';v=json.loads(p.read_text());v['CPU_seconds']=999;p.write_text(json.dumps(v))
        self.protocol['admission']['files']['APPROVED_OWNER_EXCEPTION.json']['sha256']=hashlib.sha256(p.read_bytes()).hexdigest()
        self.protocol['admission']['files']['APPROVED_OWNER_EXCEPTION.json']['bytes']=p.stat().st_size
        with self.assertRaises(ValueError):self.inspect()

    def test_inherited_validator_substitution_rejected_even_if_rehashed(self):
        p=self.root/self.protocol['admission']['validator_path'];p.write_text(p.read_text()+'\n')
        k=self.protocol['admission']['validator_path'];self.protocol['admission']['files'][k]['sha256']=hashlib.sha256(p.read_bytes()).hexdigest();self.protocol['admission']['files'][k]['bytes']=p.stat().st_size
        with self.assertRaises(ValueError):self.inspect()


if __name__=='__main__':
    if 'torch' in sys.modules:raise RuntimeError('Neural import in no-fit scope tests')
    unittest.main(verbosity=2)
