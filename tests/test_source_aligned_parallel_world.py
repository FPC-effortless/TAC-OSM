"""PAR-MEM-010 software-only regressions: no real-world capability claims."""
from __future__ import annotations
import unittest
import torch

from tac_osm.variants.source_aligned_parallel_world import SourceAlignedParallelWorld


class HistoricalReuseNoFitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(9010)
        self.m=SourceAlignedParallelWorld()
        self.x=torch.randn(2,6,14)*.1
        self.c=torch.randn(2,6,14)*.1
        self.dt=torch.tensor([[1.,2.,1.,2.,3.,1.],
                              [0.,1.,3.,1.,2.,1.]])
        self.admitted=torch.tensor([
            [True,True,False,True,False,True],
            [True,False,True,True,True,False]],dtype=torch.bool)
        self.owner=torch.tensor([31,57],dtype=torch.long)
        self.idx=torch.tensor([
            [100,101,102,103,104,105],
            [451,452,453,454,455,456]],dtype=torch.long)

    def forward_native(self,*,x=None,c=None,dt=None,admitted=None,idx=None,queries=None):
        return self.m(x if x is not None else self.x,
                      c if c is not None else self.c,
                      elapsed_dt=dt if dt is not None else self.dt,
                      write_mask=admitted if admitted is not None else self.admitted,
                      source_ids=self.owner,
                      event_index=idx if idx is not None else self.idx,
                      query_event_index=queries,
                      return_trace=True)

    def test_size_active_params_and_full_state_memory_accounting(self):
        c=self.m.contract()
        self.assertGreaterEqual(c["learned_trainable_parameters"],95000)
        self.assertLessEqual(c["learned_trainable_parameters"],105000)
        self.assertEqual(c["learned_trainable_parameters"],c["deployed_parameters"])
        self.assertEqual(c["original_time_aware_fp32"],154)
        self.assertEqual(c["added_source_aligned_fp32"],256)
        self.assertEqual(c["total_fp32_per_stream"],410)
        self.assertEqual(c["owner_and_last_event_int64_per_stream"],2)
        self.assertFalse(c["trained_source_model"])

    def test_shared_writer_reader_address_identity_and_no_content_oracle(self):
        a,p=self.m._keys(self.idx,torch.float32)
        b,q=self.m._keys(self.idx,torch.float32)
        self.assertTrue(torch.equal(a,b))
        self.assertTrue(torch.equal(p,q))
        # changing the measurement does not change any historical KEY.
        x=self.x.clone()
        x[:,2]+=200
        _,before=self.forward_native()
        _,after=self.forward_native(x=x)
        torch.testing.assert_close(before["aligned_write_key"],
                                   after["aligned_write_key"],rtol=0,atol=0)
        self.assertTrue(torch.equal(before["aligned_write_key"],after["aligned_write_key"]))

    def test_batch_scan_equals_independent_source_owned_stream(self):
        with torch.no_grad():
            self.m.predictive_memory_delta.weight.normal_(0,.1)
        batched,trace=self.forward_native()
        st=self.m.init_stream(self.owner)
        preds=[]
        for i in range(self.x.shape[1]):
            p,st,local=self.m.stream_step(
                st,self.x[:,i],self.c[:,i],
                elapsed_dt=self.dt[:,i],write_mask=self.admitted[:,i],
                source_ids=self.owner,event_index=self.idx[:,i],return_trace=True)
            preds.append(p)
            for name in ["aligned_memory","aligned_old_read","aligned_write_key",
                         "aligned_write_route","aligned_retention","aligned_strength"]:
                torch.testing.assert_close(
                    local[name],trace[name][:,i],atol=5e-5,rtol=3e-4,
                    msg="batch/stream source memory mismatch: "+name)
        torch.testing.assert_close(torch.stack(preds,1),batched,atol=5e-5,rtol=3e-4)
        self.assertEqual(st.aligned_memory.shape,(2,2,16,8))
        torch.testing.assert_close(
            st.aligned_memory,trace["aligned_memory"][:,-1],atol=5e-5,rtol=3e-4)
        self.assertEqual(st.source_owner.tolist(),[31,57])
        self.assertEqual(st.last_event_ordinal.tolist(),[105,456])

    def test_read_only_historical_query_does_not_change_state(self):
        s=self.m.init_stream(self.owner)
        for i in range(6):
            _,s=self.m.stream_step(
                s,self.x[:,i],self.c[:,i],elapsed_dt=self.dt[:,i],
                write_mask=self.admitted[:,i],source_ids=self.owner,
                event_index=self.idx[:,i])
        memory=s.aligned_memory.detach().clone()
        source_state=s.native.memory.detach().clone()
        result=self.m.historic_query(
            s,source_ids=self.owner,requested_event_index=self.idx[:,0])
        self.assertEqual(result.shape,(2,14))
        self.assertTrue(torch.isfinite(result).all())
        self.assertTrue(torch.equal(s.aligned_memory,memory))
        self.assertTrue(torch.equal(s.native.memory,source_state))
        self.assertTrue(torch.equal(
            result,self.m.historic_query(
                s,source_ids=self.owner,requested_event_index=self.idx[:,0])))

    def test_source_non_admission_is_identity_for_source_aligned_matrix(self):
        st=self.m.init_stream(self.owner)
        initial=st.aligned_memory.detach().clone()
        none=torch.zeros(2,dtype=torch.bool)
        _,out=self.m.stream_step(
            st,self.x[:,0],self.c[:,0],elapsed_dt=self.dt[:,0],
            write_mask=none,source_ids=self.owner,event_index=self.idx[:,0])
        self.assertTrue(torch.equal(out.aligned_memory,initial))
        self.assertTrue(torch.equal(out.native.memory,st.native.memory))
        self.assertEqual(out.native.count,1)
        self.assertFalse(out.native.write_seen.any())

    def test_zero_initial_predictive_memory_retains_previous_model_forecast(self):
        raw=self.m.backbone(self.x,self.c,elapsed_dt=self.dt,
                             write_mask=self.admitted)
        updated,_=self.forward_native()
        self.assertTrue(torch.equal(raw,updated))

    def test_causality_future_source_changes_cannot_poison_earlier_output(self):
        a,_=self.forward_native()
        newer=self.x.clone()
        newer[:,4:]+=1e3
        mask=self.admitted.clone()
        mask[:,4:]=~mask[:,4:]
        idx=self.idx.clone()
        idx[:,4:]+=1000
        b,_=self.forward_native(x=newer,admitted=mask,idx=idx)
        torch.testing.assert_close(a[:,:4],b[:,:4],rtol=0,atol=1e-6)

    def test_external_query_is_genuine_source_time_only_not_writer_target(self):
        # Alternate historical event request changes legal read direction;
        # source output target is never accepted as a query argument.
        chosen=torch.stack((self.idx[:,0],)*6,dim=1)
        _,t=self.forward_native(queries=chosen[:,:0] if False else None)
        with self.assertRaises(ValueError):
            self.forward_native(queries=chosen) # at t0 requested current index
        chosen=self.idx-1
        a,ta=self.forward_native(queries=chosen)
        self.assertEqual(a.shape,self.x.shape)
        self.assertTrue(torch.isfinite(ta["aligned_old_read"]).all())

    def test_causal_index_owner_validation_and_monotonic_failure_closed(self):
        bad=self.idx.clone()
        bad[:,3]=bad[:,2]
        with self.assertRaises(ValueError):
            self.forward_native(idx=bad)
        with self.assertRaises(ValueError):
            self.forward_native(queries=self.idx)
        with self.assertRaises(ValueError):
            self.forward_native(queries=self.idx+1)
        s=self.m.init_stream(self.owner)
        with self.assertRaises(ValueError):
            self.m.historic_query(s,source_ids=self.owner,
                                  requested_event_index=self.idx[:,0])
        with self.assertRaises(ValueError):
            self.m.stream_step(
                s,self.x[:,0],self.c[:,0],elapsed_dt=self.dt[:,0],
                write_mask=self.admitted[:,0],
                source_ids=torch.tensor([31,58],dtype=torch.long),
                event_index=self.idx[:,0])

    def test_trainability_end_to_end_historical_content_and_forecast_no_fit(self):
        p,trace=self.forward_native()
        # Constructed targets are ONLY differentiability fixtures; not a
        # trained or real-source retrieval result.
        loss=p.square().mean()+self.m.historical_content(
            trace["aligned_old_read"][:,1:]).square().mean()
        loss.backward()
        for name in ["source_key_rotations","source_content.weight",
                     "write_router.weight","read_router.weight",
                     "retention.weight","write_strength.weight",
                     "historical_content.weight",
                     "predictive_memory_delta.weight"]:
            tensor=dict(self.m.named_parameters())[name]
            self.assertIsNotNone(tensor.grad,name)
            self.assertTrue(torch.isfinite(tensor.grad).all(),name)
        self.assertGreater(
            dict(self.m.named_parameters())["source_content.weight"].grad.abs().sum().item(),0)

if __name__=="__main__":
    unittest.main()
