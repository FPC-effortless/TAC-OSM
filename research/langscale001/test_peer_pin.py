"""Transport-mocked pin/equality regression tests; no network or model."""
import copy,io,json,runpy,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]

class PeerPinTest(unittest.TestCase):
 def invoke(self,*,with_pin=True,bad_pin=False,wrong_peer=False):
  with tempfile.TemporaryDirectory(prefix='langscale001-peer-') as tmp:
   root=Path(tmp);scripts=root/'research/scripts';scripts.mkdir(parents=True)
   file=scripts/'check.py';file.write_text((ROOT/'research/scripts/check_pnds_remote_parity.py').read_text())
   local={'variants':[{'id':'preserved'}],'prospective_studies':{'LANGSCALE001':{'pinned':True}}}
   (root/'research/pnds-variant-math-links.json').write_text(json.dumps(local))
   sha='a'*40
   if with_pin:
    d=root/'research/langscale001';d.mkdir()
    (d/'PNDS_PIN.json').write_text(json.dumps({'repository':'FPC-effortless/pnds-lean','registry_commit':'moving-branch' if bad_pin else sha}))
   peer=copy.deepcopy(local)
   if wrong_peer:peer['prospective_studies']['LANGSCALE001']['pinned']=False
   calls=[]
   def get(url,timeout):calls.append(url);return io.BytesIO(json.dumps(peer).encode())
   with patch('urllib.request.urlopen',side_effect=get):runpy.run_path(str(file),run_name='__main__')
   return calls
 def test_immutable_pin_is_used(self):self.assertIn('/'+'a'*40+'/',self.invoke()[0])
 def test_mismatch_is_rejected(self):
  with self.assertRaisesRegex(SystemExit,'drifted'):self.invoke(wrong_peer=True)
 def test_moving_pin_is_rejected(self):
  with self.assertRaisesRegex(SystemExit,'Invalid immutable'):self.invoke(bad_pin=True)
 def test_historical_main_fallback(self):self.assertIn('/main/',self.invoke(with_pin=False)[0])
if __name__=='__main__':unittest.main()
