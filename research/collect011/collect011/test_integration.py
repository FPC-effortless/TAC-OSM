"""Engineering checks only: no model optimizer and no claim of fitted quality."""
import copy
from dataclasses import replace
from pathlib import Path
import unittest
import torch
from torch.nn import functional as F
from .memory import PhaseMemory, PhaseState, write_step
from .model import CollectiveTAC, CollectState, FFN_DIM, named_state_tensors

torch.set_num_threads(1)
FFN=FFN_DIM

def model(arm,seed=1,dtype=torch.float64):
    torch.manual_seed(seed)
    return CollectiveTAC(arm,ffn_dim=FFN[arm]).to(dtype)

def tensors(state):
    return named_state_tensors(state)


class OperatorTests(unittest.TestCase):
    def test_support_lane_no_false_negatives_and_reset(self):
        from .support import SupportLane
        a=SupportLane();b=SupportLane()
        keys=[f'owner-a-key-{i}'.encode() for i in range(128)]
        for k in keys:a.add(k)
        self.assertTrue(all(a.contains(k) for k in keys))
        self.assertTrue(all(not b.contains(k) for k in keys))
        self.assertEqual(a.state_bytes(),512)
        a.reset();self.assertTrue(all(not a.contains(k) for k in keys))
        # Deliberate saturation demonstrates the documented false-positive limit.
        tiny=SupportLane(bits=8,hashes=4)
        for k in keys:tiny.add(k)
        self.assertTrue(tiny.contains(b'unseen-key'))

    def test_validated_event_parser_is_causal_and_rejects_malformed_records(self):
        from .commit_repair_probe import syntax_commit
        chars='#?:;ABCDEFGHabcdefghijklmnop';enc={c:i for i,c in enumerate(chars)}
        def encode(s):return torch.tensor([[enc[c] for c in s]])
        text='ABC:a;?ABC:;ABC?b;ABa:c;DEF:p;'
        x=encode(text);g=syntax_commit(x,enc)
        self.assertEqual(torch.nonzero(g[0]).flatten().tolist(),[5,len(text)-1])
        cut=15;xx=x.clone();xx[:,cut:]=enc[';']
        self.assertTrue(torch.equal(g[:,:cut],syntax_commit(xx,enc)[:,:cut]))
        self.assertFalse(bool(syntax_commit(encode('ABC:a'),enc).any()))

    def test_full_gain_overwrite(self):
        torch.manual_seed(71)
        a=torch.randn(2,6,6,dtype=torch.float64)
        p=a@a.transpose(-1,-2)+torch.eye(6,dtype=torch.float64)
        s=torch.randn(2,3,6,dtype=torch.float64)
        k=F.normalize(torch.randn(2,6,dtype=torch.float64),dim=-1)
        v=torch.randn(2,3,dtype=torch.float64)
        updated,pp=write_step(s,p,k,v,torch.ones(2,dtype=torch.float64))
        torch.testing.assert_close((updated@k[...,None]).squeeze(-1),v,atol=1e-12,rtol=1e-12)
        self.assertTrue(bool((torch.linalg.eigvalsh(pp)>0).all()))

    def test_operator_gradient(self):
        torch.manual_seed(2)
        s=torch.randn(1,2,4,dtype=torch.float64,requires_grad=True)
        p=(torch.eye(4,dtype=torch.float64)*2)[None].requires_grad_()
        k=F.normalize(torch.randn(1,4,dtype=torch.float64),dim=-1).requires_grad_()
        v=torch.randn(1,2,dtype=torch.float64,requires_grad=True)
        g=torch.tensor([.3],dtype=torch.float64,requires_grad=True)
        self.assertTrue(torch.autograd.gradcheck(lambda *a:write_step(*a),(s,p,k,v,g)))

    def test_zero_write_and_read_before_write(self):
        for hybrid in (False,True):
            m=PhaseMemory(8,8,4,2,hybrid=hybrid).double()
            x=torch.randn(2,12,8,dtype=torch.float64)
            old=m.start(("a","b"));snap=old.memory.clone()
            y,new=m(x,state=old,owners=("a","b"),write_mask=torch.zeros(2,12,dtype=torch.bool))
            self.assertTrue(torch.equal(y,torch.zeros_like(y)))
            self.assertTrue(torch.equal(new.memory,snap))
            self.assertTrue(torch.equal(new.precision,old.precision))
            first,_=m(x[:,:1]);self.assertTrue(torch.equal(first,torch.zeros_like(first)))

    def test_bad_state_and_precision_rejected(self):
        m=PhaseMemory(8,8,4).double();x=torch.randn(2,4,8,dtype=torch.float64)
        s=m.start(("a","b"))
        for bad in (replace(s,owners=("b","a")),replace(s,clock=-1),replace(s,signature="x"),
                    replace(s,precision=-s.precision),replace(s,memory=s.memory.float()),
                    replace(s,memory=s.memory[:1])):
            with self.assertRaises(ValueError):m(x,state=bad,owners=("a","b"))
        with self.assertRaises(ValueError):m(x,state=s)
        with self.assertRaises(ValueError):m(x,owners=("a","a"))
        with self.assertRaises(ValueError):m(x.float())
        with torch.autocast("cpu",dtype=torch.bfloat16):
            with self.assertRaises(ValueError):m(x)

    def test_chunk_attached_gradients(self):
        torch.manual_seed(4)
        m=PhaseMemory(8,8,4).double()
        x=torch.randn(2,11,8,dtype=torch.float64,requires_grad=True)
        y,s=m(x,owners=("a","b"));g=torch.autograd.grad(y.square().sum()+s.memory.square().sum(),x)[0]
        xx=x.detach().clone().requires_grad_()
        a,p=m(xx[:,:5],owners=("a","b"));b,q=m(xx[:,5:],state=p,owners=("a","b"))
        yy=torch.cat((a,b),1);gg=torch.autograd.grad(yy.square().sum()+q.memory.square().sum(),xx)[0]
        torch.testing.assert_close(y,yy,atol=1e-12,rtol=1e-12)
        torch.testing.assert_close(g,gg,atol=1e-10,rtol=1e-10)
        torch.testing.assert_close(s.precision,q.precision,atol=1e-12,rtol=1e-12)


class ActualTACGraphTests(unittest.TestCase):
    def test_original_attention_control_matches_source(self):
        from tacround110.model import ResearchTAC110
        m=model("attention");r=ResearchTAC110(m.config).double()
        r.load_state_dict(m.state_dict(),strict=True)
        x=torch.arange(26).reshape(2,13).long()
        a,_=m(x);b,_=r(x)
        torch.testing.assert_close(a,b,atol=0,rtol=0)

    def test_parameter_match_and_no_unused_attention_weights(self):
        counts=[]
        for arm in CollectiveTAC.ARMS:
            m=model(arm);counts.append(sum(p.numel() for p in m.parameters()))
            if arm!="attention":
                self.assertFalse(any("local_query" in n or "local_key" in n or "local_value" in n for n,_ in m.named_parameters()))
            self.assertEqual(len(m.core.blocks),4)
            self.assertTrue(hasattr(m,"sparse_memory"))
        self.assertLess((max(counts)-min(counts))/min(counts),.01)

    def test_full_graph_causal_streaming_owner_isolation(self):
        x=torch.arange(34).reshape(2,17).long();owners=("alpha","beta")
        for arm in CollectiveTAC.ARMS:
            with self.subTest(arm=arm),torch.no_grad():
                m=model(arm);y,s=m(x,owners=owners)
                a,p=m(x[:,:7],owners=owners);b,q=m(x[:,7:],state=p,owners=owners)
                torch.testing.assert_close(y,torch.cat((a,b),1),atol=1e-10,rtol=1e-10)
                steps=[];ss=m.start(owners)
                for t in range(x.shape[1]):
                    out,ss=m.step(x[:,t],state=ss,owners=owners);steps.append(out)
                torch.testing.assert_close(y,torch.cat(steps,1),atol=1e-10,rtol=1e-10)
                for n,v in tensors(s).items():torch.testing.assert_close(v,tensors(q)[n],atol=1e-10,rtol=1e-10)
                poison=x.clone();poison[:,7:]=(poison[:,7:]+119)%256
                z,_=m(poison,owners=owners)
                torch.testing.assert_close(y[:,:7],z[:,:7],atol=0,rtol=0)
                other=x.clone();other[1]=(other[1]+71)%256
                z,_=m(other,owners=owners);torch.testing.assert_close(y[0],z[0],atol=0,rtol=0)
                single,_=m(x[:1],owners=owners[:1]);torch.testing.assert_close(y[:1],single,atol=1e-10,rtol=1e-10)
                with self.assertRaises(ValueError):m(x[:,:1],state=s,owners=owners[::-1])
                restore=CollectState.from_payload(s.payload());aa,_=m(x[:,:1],state=s,owners=owners);bb,_=m(x[:,:1],state=restore,owners=owners)
                torch.testing.assert_close(aa,bb,atol=0,rtol=0)

    def test_real_train_bytes_all_parameters_active_and_chunk_gradients(self):
        data=(Path(__file__).resolve().parents[1]/"data/train.bin").read_bytes()
        # This is input provenance for a no-update engineering check, not a fit.
        v=torch.tensor(list(data[:66]),dtype=torch.long).reshape(2,33)
        x,target=v[:,:-1],v[:,1:]
        for arm in CollectiveTAC.ARMS:
            with self.subTest(arm=arm):
                m=model(arm);before={n:p.detach().clone() for n,p in m.named_parameters()}
                y,s=m(x);loss=F.cross_entropy(y.flatten(0,1),target.flatten())
                loss.backward()
                for n,p in m.named_parameters():
                    self.assertIsNotNone(p.grad,n);self.assertTrue(bool(torch.isfinite(p.grad).all()),n)
                    self.assertGreater(float(p.grad.abs().sum()),0,n)
                    self.assertTrue(torch.equal(p,before[n]),n)
                grads={n:p.grad.clone() for n,p in m.named_parameters()};m.zero_grad(set_to_none=True)
                a,p=m(x[:,:13],owners=("a","b"));b,_=m(x[:,13:],state=p,owners=("a","b"))
                F.cross_entropy(torch.cat((a,b),1).flatten(0,1),target.flatten()).backward()
                for n,p in m.named_parameters():torch.testing.assert_close(p.grad,grads[n],atol=2e-9,rtol=1e-7,msg=n)

    def test_phase_state_bounded_and_detach_explicit(self):
        for arm in CollectiveTAC.ARMS[1:]:
            m=model(arm);owners=("a",)
            with torch.no_grad():
                _,s=m(torch.arange(9)[None],owners=owners)
                _,long=m(torch.arange(40)[None],state=s,owners=owners)
            self.assertEqual(s.state_bytes(),long.state_bytes())
            self.assertTrue(all(l.recent_key.numel()==0 and l.recent_value.numel()==0 for l in long.native.layers))
            _,s=m(torch.arange(4)[None],owners=owners)
            self.assertTrue(s.phase[0].memory.requires_grad)
            self.assertFalse(s.detach().phase[0].memory.requires_grad)
            for old,new in zip(s.phase,s.detach().phase):self.assertTrue(torch.equal(old.memory,new.memory))


if __name__=="__main__":unittest.main()
