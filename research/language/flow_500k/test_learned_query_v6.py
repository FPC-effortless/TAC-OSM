"""GRAPH010 candidate tests. Constructed source strings are unit fixtures only,
NEVER substitute for real FIT/independent corpus performance benchmarks.
"""
import unittest
import torch
import torch.nn.functional as F
from learned_context_query_v6 import LearnedContextQuery
from bayesian_cache_scan_v5 import ScannedCountCache

class LearnedQueryContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)
    def setUp(self):
        torch.manual_seed(202610)
        self.q=LearnedContextQuery(torch.ones(128)/128,capacity=24)
        self.assertEqual(self.q.learned_parameters(),2816)

    def test_exact_source_cache_hit_matches_and_query_abstains(self):
        x=torch.randint(0,12,(2,67),dtype=torch.long)
        a=self.q.predict_window(x)
        slow,_=ScannedCountCache(24).scan_counts(x,source_ids=("A","B"))
        self.assertTrue(torch.equal(a["exact_hit"],slow["totals"][:,:,1]>0))
        self.assertFalse(bool((a["hit"]&a["exact_hit"]).any()))
        self.assertTrue(torch.allclose(a["posterior"].sum(-1),
                                       torch.ones_like(a["hit"],dtype=torch.float32),atol=2e-6))
        self.assertTrue(bool((a["posterior"]>0).all()))

    def test_differentiable_nonexact_query_and_gradient_nonzero(self):
        x=(torch.arange(43).reshape(1,-1)*7%127).long()
        result=self.q.predict_window(x)
        self.assertGreater(int(result["hit"].sum()),0)
        # Teacher-forced next symbol, only on nonexact graph009 misses.
        selection=result["hit"][:,:-1]
        logp=result["posterior"][:,:-1].clamp_min(1e-30).log()
        loss=-logp.gather(-1,x[:,1:,None]).squeeze(-1)[selection].mean()
        grads=torch.autograd.grad(loss,(self.q.embedding.weight,self.q.project.weight))
        self.assertTrue(all(torch.isfinite(g).all() for g in grads))
        self.assertGreater(float(grads[0].norm()),1e-10)
        self.assertGreater(float(grads[1].norm()),1e-10)

    def test_chunk_parity_global_ordinal_owner_and_bounded_state(self):
        x=torch.randint(0,8,(2,89),dtype=torch.long)
        self.q.eval()
        with torch.no_grad():
            oracle=self.q.predict_window(x)
            full,state=self.q.scan(x,source_ids=("owner-A","owner-B"))
            self.assertTrue(torch.equal(full["hit"],oracle["hit"]))
            self.assertTrue(torch.equal(full["exact_hit"],oracle["exact_hit"]))
            self.assertTrue(torch.allclose(full["posterior"],oracle["posterior"],atol=3e-6,rtol=3e-6))
            for cut in (1,3,12,41,62):
                p,left=self.q.scan(x[:,:cut],source_ids=("owner-A","owner-B"))
                q,right=self.q.scan(x[:,cut:],start=cut,source_ids=("owner-A","owner-B"),initial_state=left)
                for field in ("posterior","hit","exact_hit"):
                    joined=torch.cat((p[field],q[field]),dim=1)
                    if field=="posterior":
                        self.assertTrue(torch.allclose(joined,full[field],atol=3e-6,rtol=3e-6))
                    else:self.assertTrue(torch.equal(joined,full[field]))
                self.assertEqual(right["count"],x.shape[1])
                self.assertTrue(all(len(row)<=25 for row in right["events"]))
        with self.assertRaisesRegex(ValueError,"owner"):
            self.q.scan(x[:,12:],start=12,source_ids=("owner-X","owner-B"),initial_state=left)

    def test_future_poison_does_not_affect_prefix(self):
        x=torch.randint(0,11,(1,62))
        y=x.clone();y[:,35:]=(y[:,35:]+5)%11
        with torch.no_grad():
            a=self.q.predict_window(x)
            b=self.q.predict_window(y)
            c,_=self.q.scan(x,source_ids=("owner-A",))
            d,_=self.q.scan(y,source_ids=("owner-A",))
        for field in ("posterior","hit","exact_hit"):
            self.assertTrue(torch.allclose(a[field][:,:35].float(),
                                           b[field][:,:35].float(),atol=1e-6))
            self.assertTrue(torch.allclose(c[field][:,:35].float(),
                                           d[field][:,:35].float(),atol=1e-6))

    def test_bad_input_and_forged_carry_fail_closed(self):
        x=torch.randint(0,128,(1,12))
        with self.assertRaises(ValueError):
            LearnedContextQuery(torch.zeros(128))
        with self.assertRaises(ValueError):
            self.q.scan(x,start=300,source_ids=("A",))
        _,s=self.q.scan(x,source_ids=("A",))
        with self.assertRaisesRegex(ValueError,"owner"):
            self.q.scan(x,start=12,source_ids=("B",),initial_state=s)
        bad=dict(s);bad["count"]=11
        with self.assertRaisesRegex(ValueError,"ordinal"):
            self.q.scan(x,start=12,source_ids=("A",),initial_state=bad)
        future=dict(s);future["events"]=(((16,(1,2,3),4),),)
        with self.assertRaisesRegex(ValueError,"causal"):
            self.q.scan(x,start=12,source_ids=("A",),initial_state=future)

if __name__=="__main__":unittest.main()
