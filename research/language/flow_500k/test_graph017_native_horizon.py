"""GRAPH017 constructed software tests, NOT real-data predictive evidence."""
import unittest
import torch
from torch.nn import functional as F
from graph017_native_horizon_tac import Graph017HorizonTAC500K
from tac_wave_ced_500k import WaveCED500K

class TestGraph017(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def test_500k_count_and_deployment(self):
        model=Graph017HorizonTAC500K()
        self.assertEqual(model.parameter_count(),512634)
        self.assertTrue(475000<=model.parameter_count()<=525000)
        self.assertEqual(sum(p.numel() for p in model.parameters() if p.requires_grad),512634)

    def test_future_marginals_normalize_and_finite(self):
        torch.manual_seed(17);model=Graph017HorizonTAC500K().eval()
        with torch.no_grad():outputs=model(torch.randint(0,128,(2,23)))
        self.assertEqual(set(outputs),{1,4,16})
        for h,logits in outputs.items():
            self.assertEqual(tuple(logits.shape),(2,23,128))
            self.assertTrue(torch.isfinite(logits).all())
            p=logits.softmax(-1)
            self.assertTrue(torch.allclose(p.sum(-1),torch.ones(2,23),atol=1e-6))

    def test_no_future_token_lookahead_for_all_horizons(self):
        torch.manual_seed(17017);model=Graph017HorizonTAC500K().eval()
        x=torch.randint(0,128,(2,36))
        with torch.no_grad():
            a=model(x);changed=x.clone();changed[:,18:]=(changed[:,18:]+29)%128;b=model(changed)
        for h in (1,4,16):
            self.assertTrue(torch.allclose(a[h][:,:18],b[h][:,:18],atol=2e-5,rtol=2e-5),h)

    def test_future_supervision_only_inside_same_real_window(self):
        x=torch.arange(129,dtype=torch.long).view(1,129)
        for h in (1,4,16):
            y=x[:,h:]
            self.assertEqual(y.numel(),129-h)
            self.assertEqual(y[0,0].item(),h)
            self.assertEqual(y[0,-1].item(),128)

    def test_all_horizon_parameters_have_gradient(self):
        torch.manual_seed(1717);model=Graph017HorizonTAC500K();batch=torch.randint(0,128,(2,33))
        z=model(batch[:,:-1])
        loss=sum(w*F.cross_entropy(z[h][:,:33-h].reshape(-1,128),batch[:,h:].reshape(-1))
                 for h,w in [(1,1.),(4,.15),(16,.15)])
        loss.backward()
        self.assertEqual([n for n,p in model.named_parameters() if p.grad is None],[])
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters()))

    def test_one_step_original_function_exact_if_original_weights_loaded(self):
        torch.manual_seed(155);orig=WaveCED500K().eval();candidate=Graph017HorizonTAC500K().eval()
        missing,unexpected=candidate.load_state_dict(orig.state_dict(),strict=False)
        self.assertEqual(unexpected,[])
        self.assertEqual(sorted(missing),sorted(
            n for n in candidate.state_dict() if n.startswith(('future4_','future16_'))))
        x=torch.randint(0,128,(2,13))
        with torch.no_grad():
            a=orig(x);b=candidate(x)[1]
        self.assertTrue(torch.equal(a,b))

if __name__=='__main__':unittest.main()
