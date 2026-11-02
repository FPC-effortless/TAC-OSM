"""Order/reference/gradient, baseline identity and actual causal safety fixtures."""
import json,unittest
import torch
from .common import ROOT,SPECS,build,verify_data,internal_gate
from .reference import whole_reference
from .assays import episode_spec,shuffle_state,state_hash
from collect012.common import build as old_build
from collect012.train import fingerprint
from collect011.model import named_state_tensors

torch.set_num_threads(1)

class Prefit(unittest.TestCase):
    def test_live_counts_initial_weights_and_gate(self):
        for arm in SPECS:
            m=build(arm,12001);old=old_build('hybrid_hard_p',12001)
            self.assertEqual(sum(p.numel() for p in m.parameters()),99971)
            self.assertEqual(fingerprint(m),fingerprint(old))
            registration=json.loads((ROOT/'registrations'/arm/'FAIR_COMPARISON.json').read_text())
            self.assertFalse(internal_gate(registration)['eligible_for_architecture_superiority'])

    def test_baseline_balance_forward_and_lm_gradients_identical(self):
        m=build('balance',12001).double();old=old_build('hybrid_hard_p',12001).double();x=torch.arange(22).reshape(2,11)
        y,s=m(x);a,t=old(x);self.assertTrue(torch.equal(y,a))
        for n,v in named_state_tensors(s).items():self.assertTrue(torch.equal(v,named_state_tensors(t)[n]))
        gy=torch.autograd.grad(y.square().mean(),tuple(m.parameters()));ga=torch.autograd.grad(a.square().mean(),tuple(old.parameters()))
        for v,w in zip(gy,ga):self.assertTrue(torch.equal(v,w))

    def test_full_reference_and_gradients(self):
        for arm in SPECS:
            for dtype in (torch.float32,torch.float64):
                m=build(arm,12001).to(dtype);x=torch.arange(22).reshape(2,11);mask=torch.ones_like(x,dtype=torch.bool);mask[:,3]=False
                y,s=m(x,write_mask=mask);r,rs=whole_reference(m,x,write_mask=mask);tol=8e-5 if dtype==torch.float32 else 1e-9
                torch.testing.assert_close(y,r,atol=tol,rtol=tol)
                for n,v in named_state_tensors(s).items():torch.testing.assert_close(v,named_state_tensors(rs)[n],atol=tol,rtol=tol)
                params=tuple(m.parameters());g=torch.autograd.grad(y.square().mean(),params,retain_graph=True);gg=torch.autograd.grad(r.square().mean(),params)
                for (n,_),v,w in zip(m.named_parameters(),g,gg):torch.testing.assert_close(v,w,atol=tol,rtol=5*tol,msg=n)

    def test_future_owner_chunk_and_branch_integrity(self):
        for arm in SPECS:
            m=build(arm,12001).double();x=torch.arange(34).reshape(2,17);owners=('one','two')
            with torch.no_grad():
                y,_=m(x);poison=x.clone();poison[:,8:]=(poison[:,8:]+80)%256
                z,_=m(poison);self.assertTrue(torch.equal(y[:,:8],z[:,:8]))
                poison=x.clone();poison[1]=(poison[1]+70)%256;z,_=m(poison);self.assertTrue(torch.equal(y[0],z[0]))
                a,s=m(x[:,:8],owners=owners);before=state_hash(s);b,t=m(x[:,8:],state=s,owners=owners)
                self.assertEqual(before,state_hash(s));torch.testing.assert_close(y,torch.cat((a,b),1),atol=1e-10,rtol=1e-10)
                r,_=whole_reference(m,x[:,8:],state=s,owners=owners);torch.testing.assert_close(b,r,atol=1e-10,rtol=1e-10)
                shuffled=shuffle_state(s);self.assertEqual(shuffled.owners,owners);self.assertEqual(shuffled.clock,s.clock)

    def test_real_byte_objective_and_direct_value_credit(self):
        data,_=verify_data();tokens=torch.tensor(list(data['train'][:258])).reshape(2,129)
        for arm in SPECS:
            m=build(arm,12001);y,_=m(tokens[:,:-1]);aux,counts=m.balancing()
            loss=torch.nn.functional.cross_entropy(y.flatten(0,1),tokens[:,1:].flatten())+m.balance_coefficient*aux
            loss.backward()
            for n,p in m.named_parameters():self.assertTrue(p.grad is not None and torch.isfinite(p.grad).all() and (p.grad!=0).any(),n)
            one=build(arm,12001);outputs=[];h=one.core.blocks[0].phase_memory.value.register_forward_hook(lambda mod,args,out:outputs.append(out))
            logits,_=one(torch.tensor([[65]]));g=torch.autograd.grad(logits.square().mean(),outputs[0],allow_unused=True)[0];h.remove()
            if SPECS[arm]['read_after']:self.assertTrue(g is not None and (g!=0).any())
            else:self.assertTrue(g is None or not (g!=0).any())

    def test_assay_no_answer_in_query_and_shapes(self):
        episodes=episode_spec();self.assertEqual(len(episodes),12)
        self.assertEqual(len({len(e['prefix']) for e in episodes}),1)
        for e in episodes:
            self.assertEqual(len(set(e['keys'])),6)
            self.assertNotEqual(e['initial'][0],e['latest'][0]);self.assertNotEqual(e['initial'][1],e['latest'][1])
            for k in e['keys'][4:]:self.assertNotIn(k,e['prefix'])

if __name__=='__main__':unittest.main()
