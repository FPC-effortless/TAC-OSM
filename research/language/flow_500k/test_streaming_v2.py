import unittest
from pathlib import Path
import sys
import torch
sys.path.insert(0,str(Path(__file__).resolve().parent))
from tac_wave_ced_500k import WaveCED500K as Original
from tac_wave_ced_500k_stream import WaveCED500K as Carried

class TestCarriedWaveCED(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(123)
        torch.set_num_threads(2)
        cls.model=Carried().eval()
        cls.original=Original().eval()
        cls.original.load_state_dict(cls.model.state_dict())

    def test_parameter_identity(self):
        self.assertEqual(self.model.parameter_count(),499290)
    def test_original_no_state_predictions(self):
        x=torch.randint(128,(2,128))
        with torch.no_grad():
            diff=(self.model(x)-self.original(x)).abs().max().item()
        self.assertLess(diff,2e-4)
    def test_chunk_forward_and_state(self):
        x=torch.randint(128,(2,128))
        with torch.no_grad():
            full,expected=self.model(x,return_state=True)
            for sizes in ((17,47,64),(1,1,62,64),(32,32,32,32),(63,65)):
                offset=0;state=None;blocks=[]
                for size in sizes:
                    p,state=self.model(x[:,offset:offset+size],start=offset,
                                       initial_state=state,return_state=True)
                    blocks.append(p);offset+=size
                self.assertLess((torch.cat(blocks,1)-full).abs().max().item(),2e-3)
                for key in ('basis','native','aligned','waves','concepts'):
                    self.assertLess((state[key]-expected[key]).abs().max().item(),3e-3)
                self.assertEqual(state['count'],128)
    def test_masked_chunk_forward(self):
        x=torch.randint(128,(2,96));mask=torch.rand(2,96)>.3
        with torch.no_grad():
            full,_=self.model(x,write_mask=mask,return_state=True)
            p,state=self.model(x[:,:37],write_mask=mask[:,:37],return_state=True)
            q,_=self.model(x[:,37:],start=37,write_mask=mask[:,37:],
                           initial_state=state,return_state=True)
        self.assertLess((torch.cat((p,q),1)-full).abs().max().item(),2e-3)
    def test_future_token_blindness(self):
        x=torch.randint(128,(1,128));y=x.clone();y[:,72:]=(y[:,72:]+31)%128
        with torch.no_grad():
            diff=(self.model(x)[:,:72]-self.model(y)[:,:72]).abs().max().item()
        self.assertLess(diff,1e-4)
    def test_reject_unowned_offset(self):
        with self.assertRaises(ValueError):
            self.model(torch.randint(128,(1,2)),start=33)
    def test_reject_wrong_state_count(self):
        x=torch.randint(128,(1,16));_,state=self.model(x,return_state=True)
        with self.assertRaises(ValueError):
            self.model(x,start=15,initial_state=state)
    def test_256_token_chunk_carry(self):
        x=torch.randint(128,(1,256))
        with torch.no_grad():
            full,_=self.model(x,return_state=True)
            p,state=self.model(x[:,:128],return_state=True)
            q,state=self.model(x[:,128:],start=128,initial_state=state,return_state=True)
        self.assertEqual(state['count'],256)
        self.assertLess((full-torch.cat((p,q),1)).abs().max().item(),2e-3)
    def test_chunk_gradient(self):
        from torch.nn import functional as F
        torch.manual_seed(17001)
        a=Carried();b=Carried();b.load_state_dict(a.state_dict())
        x=torch.randint(128,(1,40));target=torch.randint(128,(1,40))
        full=a(x);F.cross_entropy(full.reshape(-1,128),target.reshape(-1)).backward()
        p,state=b(x[:,:17],return_state=True)
        q,state=b(x[:,17:],start=17,initial_state=state,return_state=True)
        joined=torch.cat((p,q),dim=1)
        self.assertLess((full-joined).abs().max().item(),2e-4)
        F.cross_entropy(joined.reshape(-1,128),target.reshape(-1)).backward()
        for name in ('embedding.weight','ffn_up.weight','concept_q.weight','native_key.weight',
                     'wave_input.weight','concept_gate.weight'):
            da=dict(a.named_parameters())[name].grad
            db=dict(b.named_parameters())[name].grad
            self.assertLess((da-db).abs().max().item(),2e-4,name)

if __name__=='__main__':
    unittest.main(verbosity=2)
