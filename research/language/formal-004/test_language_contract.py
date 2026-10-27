"""TAC-LANG-FORMAL-004: original-source numerical falsifiers; not IEEE proofs."""
import unittest
import torch
from parallel_tac_language_gate_original import (
    ParallelTACLanguage, rectangular_schedule, rectangular_affine_scan)
from language_contract import (
    CheckedLanguageSession, validate_model, require_ordinal_range,
    FORCED_KEY_PERIOD, EXPECTED_FLOAT_STATE, running_mean_update)

class LanguageContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(13)
        cls.model = ParallelTACLanguage(
            decoder=75, post=10, position_mode="multiscale",
            gate_mode="silu").eval()

    def test_trained_geometry_not_constructor_default(self):
        validate_model(self.model)
        with self.assertRaisesRegex(ValueError,"one-lag"):
            validate_model(ParallelTACLanguage(decoder=75,post=10,position_mode="multiscale_relative",gate_mode="silu"))
        self.assertEqual(EXPECTED_FLOAT_STATE, 408)
        with self.assertRaises(ValueError):
            validate_model(ParallelTACLanguage())

    def test_all_named_tensor_shapes_are_covered(self):
        from language_contract import EXPECTED_PARAMETER_SHAPES
        self.assertEqual(len(EXPECTED_PARAMETER_SHAPES),36)
        self.assertEqual({name:tuple(value.shape)
            for name,value in self.model.named_parameters()},
            EXPECTED_PARAMETER_SHAPES)
        self.assertEqual(sum(p.numel() for p in self.model.parameters()),100680)
        # Same total-parameter count is NOT a sufficient typed-graph contract.
        replacement=ParallelTACLanguage(decoder=75,post=10,position_mode="multiscale")
        replacement.time_proj.weight=torch.nn.Parameter(torch.zeros(24,14))
        with self.assertRaisesRegex(ValueError,"tensor shapes"):
            validate_model(replacement)

    def test_forced_ordinal_cycle_and_boundary(self):
        self.assertEqual(FORCED_KEY_PERIOD, 155701)
        for ordinal in (0, 127, FORCED_KEY_PERIOD):
            for p in (7, 13, 29, 59):
                self.assertEqual(ordinal % p, (ordinal+FORCED_KEY_PERIOD)%p)
        require_ordinal_range(FORCED_KEY_PERIOD-1, 1)
        for start, count in ((FORCED_KEY_PERIOD, 1),(FORCED_KEY_PERIOD-1,2),(-1,1),(0,0)):
            with self.assertRaises(ValueError):
                require_ordinal_range(start,count)

    def test_batch_stream_causality(self):
        for n in (1,7,19,33):
            tokens=torch.randint(0,128,(2,n),dtype=torch.int64)
            with torch.no_grad():
                batch=self.model(tokens)
                session=CheckedLanguageSession(
                    self.model,torch.tensor([101,202],dtype=torch.int64))
                stream=torch.stack([session.step(tokens[:,t]) for t in range(n)],1)
                self.assertLess((batch-stream).abs().max().item(),1e-4)
                self.assertTrue(torch.allclose(torch.softmax(batch,-1).sum(-1),torch.ones(2,n),atol=1e-6))
                suffix=torch.randint(0,128,(2,4),dtype=torch.int64)
                extended=self.model(torch.cat([tokens,suffix],1))
                self.assertLess((batch-extended[:,:n]).abs().max().item(),1e-4)

    def test_invalid_token_owner_and_counter(self):
        session=CheckedLanguageSession(self.model,torch.tensor([2],dtype=torch.int64))
        for token in (-1,128):
            with self.assertRaisesRegex(ValueError,"vocabulary"):
                session.step(torch.tensor([token],dtype=torch.int64))
        session.state["owner"]=torch.tensor([3],dtype=torch.int64)
        with self.assertRaisesRegex(ValueError,"owner"):
            session.step(torch.tensor([1],dtype=torch.int64))
        session.state["owner"]=torch.tensor([2],dtype=torch.int64)
        session.state["count"]=FORCED_KEY_PERIOD
        with self.assertRaisesRegex(ValueError,"period"):
            session.step(torch.tensor([1],dtype=torch.int64))

    def test_stable_log_probabilities_under_fp32_underflow(self):
        logits=torch.zeros((1,128),dtype=torch.float32)
        logits[0,0]=-1000.0
        self.assertEqual(torch.softmax(logits,-1)[0,0].item(),0.)
        self.assertTrue(torch.isneginf(torch.log(torch.softmax(logits,-1))[0,0]))
        owner=torch.tensor([41],dtype=torch.int64)
        session=CheckedLanguageSession(self.model,owner)
        # Directly supply hostile but finite logits through the checked adapter:
        # the source model itself stays unchanged, no model fit needed.
        session.step=lambda token: logits
        logp=session.log_categorical(torch.tensor([0],dtype=torch.int64))
        self.assertTrue(torch.isfinite(logp).all())
        self.assertGreater(-logp[0,0].item(),1000.)
        self.assertLess(torch.logsumexp(logp,-1).abs().max().item(),1e-5)

    def test_log_categorical_real_stream_advances_once_and_backprop(self):
        model=ParallelTACLanguage(
            decoder=75,post=10,position_mode="multiscale",gate_mode="silu")
        session=CheckedLanguageSession(model,torch.tensor([37],dtype=torch.int64))
        logp=session.log_categorical(torch.tensor([9],dtype=torch.int64))
        self.assertEqual(session.state["count"],1)
        self.assertEqual(tuple(logp.shape),(1,128))
        self.assertTrue(torch.isfinite(logp).all())
        self.assertLess(torch.logsumexp(logp,-1).abs().max().item(),1e-5)
        gradient=torch.autograd.grad(-logp[0,10],model.embedding.weight)[0]
        self.assertTrue(torch.isfinite(gradient).all())

    def test_per_event_fp32_accumulator_error_receipts(self):
        from observe_fp32_sum_error import replay_fp32_additive_trace
        for steps in (1,17,257,1024):
            sample=torch.arange(steps*88,dtype=torch.float64).reshape(steps,88)
            basis=torch.sin(sample/27.0)*0.999
            receipt=replay_fp32_additive_trace(basis)
            self.assertEqual(receipt["positions"],steps)
            self.assertEqual(receipt["dim"],88)
            self.assertLessEqual(receipt["max_final_error"],
                receipt["maximum_certified_trace_bound"]+1e-11)
        with self.assertRaisesRegex(ValueError,"FP64"):
            replay_fp32_additive_trace(torch.ones(2,88,dtype=torch.float32))

    def test_rectangular_scan_matches_serial_corrective_write(self):
        generator=torch.Generator().manual_seed(9)
        for dtype,tol in ((torch.float64,1e-11),(torch.float32,2e-5)):
            k=torch.nn.functional.normalize(torch.randn(2,12,8,generator=generator,dtype=dtype),dim=-1)
            v=torch.tanh(torch.randn(2,12,16,generator=generator,dtype=dtype))
            alpha=torch.sigmoid(torch.randn(2,12,1,generator=generator,dtype=dtype))
            beta=torch.sigmoid(torch.randn(2,12,1,generator=generator,dtype=dtype))
            P,Q=rectangular_schedule(k,v,alpha,beta)
            parallel=rectangular_affine_scan(P,Q)
            m=torch.zeros(2,16,8,dtype=dtype)
            states=[]
            for t in range(12):
                read=(m@k[:,t].unsqueeze(-1)).squeeze(-1)
                m=alpha[:,t].unsqueeze(-1)*m+(
                    beta[:,t]*(v[:,t]-alpha[:,t]*read)
                    ).unsqueeze(-1)*k[:,t].unsqueeze(-2)
                states.append(m)
            self.assertLess((parallel-torch.stack(states,1)).abs().max().item(),tol)

    def test_scan_backward_matches_serial_autograd(self):
        torch.manual_seed(28)
        x=[torch.randn(2,7,d,dtype=torch.float64,requires_grad=True)
           for d in (8,16,1,1)]
        def evaluate(parallel):
            k=torch.nn.functional.normalize(x[0],dim=-1)
            v=x[1].tanh()
            a=x[2].sigmoid()
            b=x[3].sigmoid()
            P,Q=rectangular_schedule(k,v,a,b)
            if parallel:return rectangular_affine_scan(P,Q).square().sum()
            m=Q.new_zeros(2,16,8)
            states=[]
            for t in range(7):
                m=m@P[:,t]+Q[:,t]
                states.append(m)
            return torch.stack(states,1).square().sum()
        p=evaluate(True);q=evaluate(False)
        self.assertLess(abs(p.item()-q.item()),1e-10)
        gp=torch.autograd.grad(p,x)
        gq=torch.autograd.grad(q,x)
        for a,b in zip(gp,gq):
            self.assertLess((a-b).abs().max().item(),1e-9)

    def test_alternative_running_mean_bound(self):
        torch.manual_seed(7)
        x=torch.tanh(torch.randn(257,88,dtype=torch.float64))
        m=torch.zeros(88,dtype=torch.float64)
        for i in range(len(x)):
            m=running_mean_update(m,x[i],i)
            self.assertLessEqual(m.abs().max().item(),1+1e-10)
        self.assertLess((m-x.mean(0)).abs().max().item(),1e-12)
        self.assertGreater((x.sum(0)-m).abs().max().item(),1.0)

if __name__=="__main__":
    unittest.main()
