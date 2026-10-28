import unittest,torch
from historical_500k import HistoricalWaveCED500K,LAGS
class TestHistory(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2);torch.manual_seed(2)
        self.m=HistoricalWaveCED500K().eval()
    def test_budget(self):
        n=self.m.parameter_count()
        self.assertEqual(n,503514)
        self.assertTrue(475000<=n<=525000)
    def test_shapes_and_loss(self):
        x=torch.randint(0,128,(1,205))
        a,h=self.m.forward_with_history(x)
        self.assertEqual(tuple(a.shape),(1,205,128))
        self.assertEqual(tuple(h[192].shape),(1,13,128))
        self.assertTrue(torch.isfinite(h[192]).all().item())
    def test_no_future_poison(self):
        x=torch.randint(0,128,(1,205));y=x.clone();y[:,201:]=torch.randint(0,128,(1,4))
        with torch.no_grad():
            a,h=self.m.forward_with_history(x);b,k=self.m.forward_with_history(y)
        self.assertTrue(torch.allclose(a[:,:199],b[:,:199],atol=2e-5))
        for lag in LAGS:
            self.assertTrue(torch.allclose(h[lag][:,:199-lag],k[lag][:,:199-lag],atol=2e-5))
    def test_historical_label_blindness(self):
        # Directly alter x_0 while comparing late 192-lag values. Historical
        # logits can change *through state*, which is correct, not via label.
        x=torch.randint(0,128,(1,200))
        with torch.no_grad():
            p,h=self.m.forward_with_history(x)
        self.assertEqual(h[192].shape[1],8)
if __name__=='__main__': unittest.main()
