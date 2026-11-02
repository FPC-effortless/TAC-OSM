"""Whole-source prefit checks for the actual 100K graphs, no optimization."""
import hashlib,json,unittest
import torch
from .common import ROOT,SPECS,EXPECTED,build,verify_data,internal_gate
from .reference import whole_reference
from collect011.model import named_state_tensors

torch.set_num_threads(1)

class PrefitTests(unittest.TestCase):
    def test_source_gate_and_counts(self):
        _,split=verify_data();counts=[]
        for arm in SPECS:
            m=build(arm,12000);n=sum(p.numel() for p in m.parameters());counts.append(n)
            manifest=json.loads((ROOT/'registrations'/arm/'FAIR_COMPARISON.json').read_text())
            self.assertFalse(internal_gate(manifest)['eligible_for_architecture_superiority'])
            self.assertEqual(n,EXPECTED[arm])
        self.assertLess(max(counts)/min(counts)-1,.01)

    def test_full_graph_reference_values_and_gradients(self):
        for arm in SPECS:
            for dtype in (torch.float32,torch.float64):
                with self.subTest(arm=arm,dtype=dtype):
                    m=build(arm,12000).to(dtype)
                    x=torch.arange(22).reshape(2,11);mask=torch.ones_like(x,dtype=torch.bool);mask[:,3]=False
                    y,s=m(x,write_mask=mask);r,rs=whole_reference(m,x,write_mask=mask)
                    tol=4e-5 if dtype==torch.float32 else 2e-10
                    torch.testing.assert_close(y,r,atol=tol,rtol=tol)
                    for name,v in named_state_tensors(s).items():
                        torch.testing.assert_close(v,named_state_tensors(rs)[name],atol=tol,rtol=tol)
                    params=tuple(m.parameters())
                    g=torch.autograd.grad(y.square().mean(),params,retain_graph=True,allow_unused=True)
                    gr=torch.autograd.grad(r.square().mean(),params,allow_unused=True)
                    for (name,_),a,b in zip(m.named_parameters(),g,gr):
                        self.assertIsNotNone(a,name);self.assertIsNotNone(b,name)
                        torch.testing.assert_close(a,b,atol=tol,rtol=5*tol,msg=name)

    def test_causal_and_chunk_semantics(self):
        x=torch.arange(34).reshape(2,17);owners=('one','two')
        for arm in SPECS:
            m=build(arm,12000).double()
            with torch.no_grad():
                y,_=m(x);poison=x.clone();poison[:,8:]=(poison[:,8:]+120)%256
                z,_=m(poison);torch.testing.assert_close(y[:,:8],z[:,:8],atol=0,rtol=0)
                a,s=m(x[:,:8],owners=owners);b,_=m(x[:,8:],state=s,owners=owners)
                torch.testing.assert_close(y,torch.cat((a,b),1),atol=1e-10,rtol=1e-10)
                r,_=whole_reference(m,x[:,8:],state=s,owners=owners)
                torch.testing.assert_close(b,r,atol=1e-10,rtol=1e-10)

    def test_real_fit_targets_and_all_parameter_activity(self):
        data,_=verify_data();arr=torch.tensor(list(data['train'][:258])).reshape(2,129)
        for arm in SPECS:
            m=build(arm,12000);y,_=m(arr[:,:-1])
            torch.nn.functional.cross_entropy(y.flatten(0,1),arr[:,1:].flatten()).backward()
            for name,p in m.named_parameters():
                self.assertIsNotNone(p.grad,name);self.assertTrue(bool(torch.isfinite(p.grad).all()),name)
                self.assertTrue(bool((p.grad!=0).any()),name)

if __name__=='__main__':unittest.main()
