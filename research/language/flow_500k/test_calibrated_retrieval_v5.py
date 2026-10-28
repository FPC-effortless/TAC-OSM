import unittest
import torch
from bayesian_cache_scan_v5 import ScannedCountCache
from source_calibrated_tac_v5 import SourceCalibratedTAC
from tac_wave_ced_500k_v4 import SlotTAC500K

class CountContracts(unittest.TestCase):
 def setUp(self):
  torch.manual_seed(9121);torch.set_num_threads(2)
 def test_exact_naive_source_counts_and_chunk_parity(self):
  for cap in (12,32,64):
   x=torch.randint(0,5,(2,109),dtype=torch.long)
   owners=('A','B');c=ScannedCountCache(cap)
   full,_=c.scan_counts(x,source_ids=owners)
   for row in range(2):
    seq=x[row].tolist()
    for now in range(109):
     for oi,k in enumerate((2,3,4)):
      total=0;freq=[0]*128
      if now>=k-1:
       for end in range(k-1,now-1):
        q=end+1
        if now-q>cap:continue
        if seq[end-k+1:end+1]==seq[now-k+1:now+1]:
         total+=1;freq[seq[q]]+=1
      self.assertEqual(int(full['totals'][row,now,oi]),total)
      self.assertEqual(full['counts'][row,now,oi].tolist(),freq)
   for cut in (1,5,31,63,88):
    first,state=c.scan_counts(x[:,:cut],source_ids=owners)
    rest,_=c.scan_counts(x[:,cut:],start=cut,source_ids=owners,initial_state=state)
    for key in full:self.assertTrue(torch.equal(full[key],torch.cat((first[key],rest[key]),dim=1)))
    with self.assertRaisesRegex(ValueError,'owner'):
     c.scan_counts(x[:,cut:],start=cut,source_ids=('X','B'),initial_state=state)
 def test_positivity_and_frozen_model_stream(self):
  x=torch.tensor([[1,2,3,4,5,9,1,2,3,4,5,8]*9],dtype=torch.long)
  core=SlotTAC500K().eval()
  self.assertEqual(core.parameter_count(),499290)
  model=SourceCalibratedTAC(core,torch.ones(128)/128).eval()
  with torch.no_grad():
   full=model(x,source_ids=('A',))
   _,state=model(x[:,:79],source_ids=('A',),return_state=True)
   late=model(x[:,79:],start=79,source_ids=('A',),initial_state=state)
  self.assertTrue(torch.allclose(full[:,79:],late,atol=3e-3,rtol=3e-3))
  self.assertTrue(torch.allclose(full.sum(-1),torch.ones_like(full[...,0]),atol=2e-5))
  self.assertTrue(bool((full>0).all()))
 def test_future_poison_cannot_change_prior_predictions(self):
  x=torch.randint(0,10,(1,111),dtype=torch.long)
  c=ScannedCountCache(26);full,_=c.scan_counts(x,source_ids=('A',))
  altered=x.clone();altered[:,52:]=(altered[:,52:]+3)%10
  changed,_=c.scan_counts(altered,source_ids=('A',))
  for k in full:self.assertTrue(torch.equal(full[k][:,:52],changed[k][:,:52]))
if __name__=='__main__':unittest.main()
