"""PAR-TIME-001 constructed NUMERICAL tests (never empirical skill evidence)."""
import unittest
import torch

from tac_osm.variants.parallel_residual_world import ParallelResidualWorld
from tac_osm.variants.parallel_time_age_world import TimeAwareParallelResidualWorld


class TimeAgeNoFitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(271028)
        self.model=TimeAwareParallelResidualWorld()
        self.obs=torch.randn(2,5,14)*0.1
        self.act=torch.randn(2,5,14)*0.1
        self.dt=torch.tensor([[1.,2.,3.,4.,5.],[0.,1.,1.,0.,3.]])
        self.mask=torch.tensor([[True,False,False,True,False],
                                [False,False,True,False,True]],dtype=torch.bool)

    def test_parameter_budget_and_state_metadata(self):
        count=sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.assertEqual(count,96639)
        self.assertTrue(95000<=count<=105000)
        state=self.model.init_stream(2)
        self.assertEqual(state.fast_sum.shape,(2,88))
        self.assertEqual(state.memory.shape,(2,8,8))
        self.assertEqual(state.elapsed_clock.shape,(2,))
        self.assertEqual(state.last_write_clock.shape,(2,))
        self.assertEqual(state.write_seen.dtype,torch.bool)
        self.assertEqual(sum(v.numel() for v in
            (state.fast_sum,state.memory,state.elapsed_clock,state.last_write_clock))//2,154)

    def test_batch_stream_parity_mixed_source_write_masks(self):
        with torch.no_grad():
            self.model.core.time_projection.weight.fill_(0.0125)
        batched,full=self.model(self.obs,self.act,elapsed_dt=self.dt,
                                write_mask=self.mask,return_trace=True)
        stream=self.model.init_stream(2)
        outputs=[]
        for t in range(5):
            p,stream,s=self.model.stream_step(
                stream,self.obs[:,t],self.act[:,t],
                elapsed_dt=self.dt[:,t],write_mask=self.mask[:,t],
                return_trace=True)
            outputs.append(p)
            for key in ("memory","old_read","fast","slow","embedding","time_features"):
                torch.testing.assert_close(s[key],full[key][:,t],rtol=7e-4,atol=2e-4)
        torch.testing.assert_close(torch.stack(outputs,1),batched,rtol=7e-4,atol=2e-4)
        self.assertEqual(stream.count,5)
        torch.testing.assert_close(stream.memory,full["memory"][:,-1],rtol=7e-4,atol=2e-4)
        torch.testing.assert_close(stream.elapsed_clock,self.dt.sum(1),rtol=0,atol=0)
        torch.testing.assert_close(stream.last_write_clock,torch.tensor([10.,5.]),rtol=0,atol=0)
        self.assertEqual(stream.write_seen.tolist(),[True,True])

    def test_age_means_only_most_recent_previous_admitted_write(self):
        _,trace=self.model(self.obs,self.act,elapsed_dt=self.dt,write_mask=self.mask,return_trace=True)
        phi=trace["time_features"]
        # B0: before t0 no write, after t0 admitted at time1:
        # age before t1=2, t2=5, t3=9; next admitted t3(time10);
        # age before t4=5. The 8x8 matrix has no per-event timestamps.
        target=torch.tensor([1.,2.,5.,9.,5.])
        torch.testing.assert_close(phi[0,:,5].expm1(),target,rtol=1e-6,atol=1e-6)
        self.assertEqual(phi[0,:,6].tolist(),[0.,1.,1.,1.,1.])
        self.assertEqual(phi[1,:,6].tolist(),[0.,0.,0.,1.,1.])

    def test_zero_projection_equals_original_when_all_steps_write(self):
        torch.manual_seed(871)
        plain=ParallelResidualWorld()
        new=TimeAwareParallelResidualWorld()
        received=new.core.load_state_dict(plain.core.state_dict(),strict=False)
        self.assertEqual(received.unexpected_keys,[])
        self.assertEqual(received.missing_keys,["time_projection.weight"])
        new.residual_ffn.load_state_dict(plain.residual_ffn.state_dict())
        new.residual_logit.data.copy_(plain.residual_logit.data)
        target=plain(self.obs,self.act)
        actual=new(self.obs,self.act,elapsed_dt=self.dt,
                   write_mask=torch.ones_like(self.mask))
        torch.testing.assert_close(actual,target,rtol=1e-6,atol=1e-6)

    def test_temporal_path_is_learnable_and_receives_gradients(self):
        y=self.model(self.obs,self.act,elapsed_dt=self.dt,write_mask=self.mask)
        (y.square().mean()).backward()
        g=self.model.core.time_projection.weight.grad
        self.assertIsNotNone(g)
        self.assertTrue(torch.isfinite(g).all())
        self.assertGreater(g.abs().sum().item(),0)
        for name,param in self.model.named_parameters():
            if name.startswith("residual_ffn") or name=="residual_logit":
                self.assertIsNotNone(param.grad,name)
                self.assertTrue(torch.isfinite(param.grad).all(),name)

    def test_real_elapsed_time_changes_predictions_after_time_weight_enabled(self):
        with torch.no_grad():
            self.model.core.time_projection.weight.zero_()
            self.model.core.time_projection.weight[:,3]=0.25
        a=self.model(self.obs,self.act,elapsed_dt=self.dt,write_mask=self.mask)
        later=self.dt.clone()
        later[:,2] += 20.0
        b=self.model(self.obs,self.act,elapsed_dt=later,write_mask=self.mask)
        torch.testing.assert_close(a[:,:2],b[:,:2],atol=1e-6,rtol=0)
        self.assertGreater((a[:,2:]-b[:,2:]).abs().max().item(),1e-8)

    def test_future_time_and_write_mask_do_not_poison_past(self):
        a=self.model(self.obs,self.act,elapsed_dt=self.dt,write_mask=self.mask)
        dt=self.dt.clone()
        mask=self.mask.clone()
        dt[:,3:]+=900.0
        mask[:,3:]=~mask[:,3:]
        b=self.model(self.obs,self.act,elapsed_dt=dt,write_mask=mask)
        torch.testing.assert_close(a[:,:3],b[:,:3],atol=1e-6,rtol=0)

    def test_reject_invalid_timestamp_and_mask_without_mutation(self):
        before=self.model.init_stream(2)
        bad=self.dt.clone()
        bad[0,1]=-1
        with self.assertRaises(ValueError):
            self.model(self.obs,self.act,elapsed_dt=bad,write_mask=self.mask)
        bad=self.dt.clone()
        bad[1,0]=float("nan")
        with self.assertRaises(ValueError):
            self.model(self.obs,self.act,elapsed_dt=bad,write_mask=self.mask)
        with self.assertRaises(ValueError):
            self.model(self.obs,self.act,elapsed_dt=self.dt,
                       write_mask=self.mask.float())
        with self.assertRaises(TypeError):
            self.model(self.obs,self.act,write_mask=self.mask)
        with self.assertRaises(ValueError):
            self.model.stream_step(
                before,self.obs[:,0],self.act[:,0],
                elapsed_dt=self.dt[:,0].double(),write_mask=self.mask[:,0])
        torch.testing.assert_close(before.memory,torch.zeros_like(before.memory),atol=0,rtol=0)

    def test_read_without_admission_does_not_advance_matrix(self):
        state=self.model.init_stream(2)
        mask=torch.zeros(2,dtype=torch.bool)
        p,new,tr=self.model.stream_step(
            state,self.obs[:,0],self.act[:,0],
            elapsed_dt=self.dt[:,0],write_mask=mask,return_trace=True)
        torch.testing.assert_close(new.memory,state.memory,atol=0,rtol=0)
        self.assertFalse(new.write_seen.any().item())
        self.assertTrue(torch.isfinite(p).all())
        self.assertEqual(new.count,1)


if __name__=="__main__":
    unittest.main()
