"""Constructed 320-event operator tests only; NOT real memory evidence."""
import unittest
import torch
from parallel_tac_language_gate_original import ParallelTACLanguage

class LongContextSourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def test_causal_prefix_and_256_query_shape(self):
        for mode in ("silu","tanh"):
            torch.manual_seed(88)
            m=ParallelTACLanguage(decoder=75,position_mode="multiscale",gate_mode=mode).eval()
            x=torch.randint(128,(1,320))
            with torch.no_grad():
                full,tr=m(x,return_trace=True)
                prefix=m(x[:,:128])
                hist,target=m.historical_logits(x,tr,256)
            self.assertTrue(torch.equal(full[:,:128],prefix))
            self.assertEqual(tuple(hist.shape),(1,64,128))
            self.assertTrue(torch.equal(target,x[:,:64]))
            self.assertTrue(torch.isfinite(hist).all())
            self.assertEqual(sum(p.numel() for p in m.parameters()),100680)

    def test_no_invalid_historical_ordinal(self):
        m=ParallelTACLanguage(decoder=75,position_mode="multiscale",gate_mode="silu").eval()
        x=torch.randint(128,(1,320))
        with torch.no_grad():
            _,tr=m(x,return_trace=True)
            with self.assertRaises(ValueError):
                m.historical_logits(x,tr,320)
            with self.assertRaises(ValueError):
                m.historical_logits(x,tr,0)

if __name__=="__main__":unittest.main()
