import copy
import io
import unittest
from unittest.mock import patch
import numpy as np
import torch
from tac_osm.decision_sys1_mamba2_003 import (
    Mamba2RiskModel,Mamba2ReferenceBlock,MambaState,RMSNorm,count_and_state_receipt,
    official_mamba_class)
from tac_osm.decision_sys1_ssd_reference_003 import ssd_chunk_reference

torch.set_num_threads(1)

class SSDTests(unittest.TestCase):
    def setUp(self):torch.manual_seed(6101)
    def fixture(self):
        x=torch.randn(2,8,2,3,dtype=torch.float64)
        a=-torch.rand(2,8,2,dtype=torch.float64)
        b=torch.randn(2,8,2,4,dtype=torch.float64)
        c=torch.randn_like(b)
        s=torch.randn(2,2,3,4,dtype=torch.float64)
        return x,a,b,c,s
    def test_scalar_numpy_reference(self):
        x,a,b,c,s=self.fixture()
        xn,an,bn,cn,sn=[t.numpy().copy() for t in (x,a,b,c,s)]
        values=[]
        for t in range(8):
            sn=np.exp(an[:,t])[:,:,None,None]*sn+xn[:,t,:,:,None]*bn[:,t,:,None,:]
            values.append(np.sum(sn*cn[:,t,:,None,:],-1))
        for block in (1,2,4,8):
            y,final=ssd_chunk_reference(x,a,b,c,block,s)
            np.testing.assert_allclose(y.numpy(),np.stack(values,1),rtol=1e-12,atol=1e-12)
            np.testing.assert_allclose(final.numpy(),sn,rtol=1e-12,atol=1e-12)
    def test_all_operator_gradients(self):
        args=[t.clone().requires_grad_() for t in self.fixture()]
        x,a,b,c,s=args
        y,final=ssd_chunk_reference(x,a,b,c,4,s)
        loss=y.square().sum()+final.square().sum()
        grads=torch.autograd.grad(loss,args)
        args2=[t.detach().clone().requires_grad_() for t in args]
        x,a,b,c,s=args2; values=[]
        for t in range(8):
            s=s*a[:,t,:,None,None].exp()+torch.einsum('bhp,bhn->bhpn',x[:,t],b[:,t])
            values.append(torch.einsum('bhpn,bhn->bhp',s,c[:,t]))
        loss2=torch.stack(values,1).square().sum()+s.square().sum()
        for ga,gb in zip(grads,torch.autograd.grad(loss2,args2)):
            torch.testing.assert_close(ga,gb,rtol=1e-11,atol=1e-11)
    def test_operator_gradcheck(self):
        x=torch.randn(1,2,1,1,dtype=torch.float64,requires_grad=True)
        a=(-torch.rand(1,2,1,dtype=torch.float64)).requires_grad_()
        b=torch.randn(1,2,1,1,dtype=torch.float64,requires_grad=True)
        c=torch.randn_like(b,requires_grad=True)
        s=torch.randn(1,1,1,1,dtype=torch.float64,requires_grad=True)
        self.assertTrue(torch.autograd.gradcheck(lambda *p:ssd_chunk_reference(*p[:4],2,p[4]),(x,a,b,c,s)))
    def test_invalid_length_and_dtype(self):
        x,a,b,c,s=self.fixture()
        for k in (0,3):
            with self.assertRaises(ValueError):ssd_chunk_reference(x,a,b,c,k,s)
        with self.assertRaises(ValueError):ssd_chunk_reference(x,a.float(),b,c,4,s)


class MambaRiskTests(unittest.TestCase):
    def setUp(self):torch.manual_seed(6102)
    def model(self):return Mamba2RiskModel('reference').double()
    def test_exact_counts_and_state(self):
        r=count_and_state_receipt()
        self.assertEqual(r['parameters'],99857)
        self.assertEqual(r['state_scalars_per_entity'],18432)
        self.assertFalse(r['official_kernel_qualified'])
        self.assertLess((99977-99816)/99816,.01)
    def test_no_invented_backend_or_silent_fallback(self):
        with self.assertRaises(ValueError):Mamba2RiskModel('ssm')
        with patch('torch.cuda.is_available',return_value=False),self.assertRaises(RuntimeError):
            official_mamba_class()
    def test_fp64_full_step_and_final_states(self):
        m=self.model()
        for length in (1,3,16,17,33):
            x=torch.randn(2,length,212,dtype=torch.float64)
            y,s=m(x);state=m.initial(2);ys=[]
            for t in range(length):
                v,state=m.step(x[:,t],state);ys.append(v)
            torch.testing.assert_close(y,torch.stack(ys,1),rtol=2e-11,atol=2e-11)
            for a,b in zip(s,state):
                torch.testing.assert_close(a.conv,b.conv,rtol=1e-12,atol=1e-12)
                torch.testing.assert_close(a.ssm,b.ssm,rtol=1e-12,atol=1e-12)
    def test_fp32_full_step(self):
        m=Mamba2RiskModel('reference');x=torch.randn(2,17,212)
        y,_=m(x);state=m.initial(2);ys=[]
        for t in range(17):v,state=m.step(x[:,t],state);ys.append(v)
        torch.testing.assert_close(y,torch.stack(ys,1),rtol=1e-5,atol=1e-6)
    def test_all_parameter_and_input_gradients(self):
        m=self.model();other=copy.deepcopy(m)
        x=torch.randn(2,9,212,dtype=torch.float64,requires_grad=True)
        xx=x.detach().clone().requires_grad_()
        y,s=m(x);(y.square().mean()+sum(t.ssm.square().mean() for t in s)).backward()
        state=other.initial(2);ys=[]
        for t in range(9):v,state=other.step(xx[:,t],state);ys.append(v)
        (torch.stack(ys,1).square().mean()+sum(t.ssm.square().mean() for t in state)).backward()
        for (name,p),(othername,q) in zip(m.named_parameters(),other.named_parameters()):
            self.assertEqual(name,othername)
            self.assertIsNotNone(p.grad,name)
            self.assertTrue(torch.isfinite(p.grad).all(),name)
            self.assertGreater(float(p.grad.abs().max()),0,name)
            torch.testing.assert_close(p.grad,q.grad,rtol=1e-9,atol=1e-11,msg=name)
        torch.testing.assert_close(x.grad,xx.grad,rtol=1e-9,atol=1e-11)
    def test_chunk_continuation(self):
        m=self.model();x=torch.randn(2,19,212,dtype=torch.float64)
        y,s=m(x);a,sa=m(x[:,:7]);b,sb=m(x[:,7:],sa)
        torch.testing.assert_close(y,torch.cat((a,b),1),rtol=1e-11,atol=1e-11)
        for aa,bb in zip(s,sb):torch.testing.assert_close(aa.ssm,bb.ssm,rtol=1e-11,atol=1e-11)
    def test_chunk_continuation_gradients(self):
        m=self.model();other=copy.deepcopy(m)
        x=torch.randn(1,5,212,dtype=torch.float64)
        y,_=m(x);y.square().sum().backward()
        a,s=other(x[:,:2]);b,_=other(x[:,2:],s)
        torch.cat((a,b),1).square().sum().backward()
        for p,q in zip(m.parameters(),other.parameters()):torch.testing.assert_close(p.grad,q.grad,rtol=1e-9,atol=1e-11)
    def test_future_poison_and_gradient_cut(self):
        m=self.model();x=torch.randn(1,9,212,dtype=torch.float64,requires_grad=True)
        y,_=m(x); y[:,3].sum().backward()
        self.assertEqual(float(x.grad[:,4:].abs().max()),0)
        changed=x.detach().clone();changed[:,4:]=1e6
        other,_=m(changed)
        torch.testing.assert_close(y[:,:4],other[:,:4],rtol=1e-12,atol=1e-12)
    def test_entity_batch_isolation(self):
        m=self.model();x=torch.randn(2,6,212,dtype=torch.float64)
        y,_=m(x)
        for i in range(2):
            yy,_=m(x[i:i+1]);torch.testing.assert_close(y[i:i+1],yy,rtol=1e-11,atol=1e-11)
    def test_caller_state_immutable(self):
        m=self.model();s=m.initial(2);saved=copy.deepcopy(s)
        m.step(torch.randn(2,212,dtype=torch.float64),s)
        for a,b in zip(s,saved):
            self.assertTrue(torch.equal(a.conv,b.conv));self.assertTrue(torch.equal(a.ssm,b.ssm))
    def test_state_has_gradient_credit(self):
        b=Mamba2ReferenceBlock(d_model=4,d_state=3,headdim=2).double()
        state=b.initial(1)
        state=MambaState(state.conv.requires_grad_(),state.ssm.requires_grad_())
        y,_=b(torch.randn(1,3,4,dtype=torch.float64),state,block_len=2)
        y.square().sum().backward()
        self.assertGreater(float(state.conv.grad.abs().sum()),0)
        self.assertGreater(float(state.ssm.grad.abs().sum()),0)
    def test_checkpoint_and_restart(self):
        m=self.model();x=torch.randn(1,7,212,dtype=torch.float64)
        first,s=m(x[:,:3]);expected,end=m(x[:,3:],s)
        buf=io.BytesIO();torch.save(m.state_dict(),buf);buf.seek(0)
        clone=self.model();clone.load_state_dict(torch.load(buf,weights_only=True))
        with torch.no_grad():actual,_=clone(x[:,3:],s)
        self.assertTrue(torch.equal(expected,actual))
    def test_gated_norm_order(self):
        n=RMSNorm(4).double();x=torch.randn(3,4,dtype=torch.float64);z=torch.randn_like(x)
        v=x*torch.nn.functional.silu(z)
        expected=v/torch.sqrt(v.square().mean(-1,keepdim=True)+1e-5)
        torch.testing.assert_close(n(x,z),expected)
        wrong=n(x)*torch.nn.functional.silu(z)
        self.assertGreater(float((wrong-expected).detach().abs().max()),.01)
    def test_invalid_shape_dtype_and_nan(self):
        m=self.model()
        with self.assertRaises(ValueError):m(torch.randn(1,0,212,dtype=torch.float64))
        with self.assertRaises(ValueError):m(torch.randn(1,3,211,dtype=torch.float64))
        with self.assertRaises(ValueError):m(torch.full((1,3,212),float('nan'),dtype=torch.float64))
        b=Mamba2ReferenceBlock().double()
        with self.assertRaises(ValueError):b.step(torch.randn(1,64),b.initial(1))
    def test_wrong_layer_state_rejected(self):
        m=self.model()
        with self.assertRaises(ValueError):m(torch.randn(1,3,212,dtype=torch.float64),())
        with self.assertRaises(ValueError):m.step(torch.randn(1,212,dtype=torch.float64),())

if __name__=='__main__':unittest.main()
