"""No-data regression checks for the actual exploratory runner's default deny."""
import copy,json,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import exploratory_fit as runner
HERE=Path(__file__).resolve().parent
class GuardTests(unittest.TestCase):
 def fixture(self):
  a=json.loads((HERE/'EXPLORATORY_AUTHORIZATION.json').read_text());a.update(owner_authorized=True,explicit_exception=runner.PHRASE,authorization_record='CONSTRUCTED UNIT AUTHORIZATION FIXTURE; NEVER USED TO RUN FITS')
  return a
 def check(self,a):
  with tempfile.TemporaryDirectory(prefix='langscale001-guard-') as tmp:
   p=Path(tmp)/'auth.json';p.write_text(json.dumps(a));return runner.authorized(p)
 def test_default_proposal_denies(self):
  with self.assertRaisesRegex(ValueError,'BLOCKED'):runner.authorized(HERE/'EXPLORATORY_AUTHORIZATION.json')
 def test_false_record_prevents_all_model_and_data_work(self):
  args=SimpleNamespace(authorization=HERE/'EXPLORATORY_AUTHORIZATION.json')
  with self.assertRaisesRegex(ValueError,'BLOCKED'):runner.fit(args)
 def test_new_dev_access_is_rejected(self):
  a=self.fixture();a['dev_access']=True
  with self.assertRaisesRegex(ValueError,'boundary'):self.check(a)
 def test_unknown_protocol_is_rejected(self):
  a=self.fixture();a['protocol_sha256']='0'*64
  with self.assertRaisesRegex(ValueError,'protocol'):self.check(a)
 def test_missing_authorization_record_rejected(self):
  a=self.fixture();a['authorization_record']=None
  with self.assertRaisesRegex(ValueError,'record'):self.check(a)
 def test_changed_source_rejected(self):
  a=self.fixture();key=next(iter(a['code_sha256']));a['code_sha256'][key]='0'*64
  with self.assertRaisesRegex(ValueError,'code changed'):self.check(a)
 def test_valid_unit_record_does_not_trigger_training(self):
  a=self.fixture();self.assertTrue(self.check(a)['owner_authorized'])
if __name__=='__main__':unittest.main()
