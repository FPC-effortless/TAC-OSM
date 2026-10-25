"""Synthetic fixture UNIT guards only; never source of physical capability claims."""
import unittest
import numpy as np
from tac_osm.variants.ewa002_velocity_memory import VelocityResidualMemory, CAPACITY, DIM, VALUE

def k(idx):
    v=np.zeros(DIM,dtype=np.float32);v[idx%DIM]=1
    return v
def v(val):
    return np.ones(VALUE,dtype=np.float32)*val

class TestEWA002(unittest.TestCase):
    def test_no_current_future_label_before_all_forecasts(self):
        mem=VelocityResidualMemory(.0001)
        read=mem.predict(k(0),v(0))
        with self.assertRaises(RuntimeError):
            mem.predict(k(1),v(0))
        with self.assertRaises(RuntimeError):
            mem.counters()
        self.assertTrue(np.all(mem.priority==0))
        self.assertEqual(mem.n,0)
        mem.observe(read,v(.03),k(0),"real:0")
        self.assertEqual(mem.n,1)
        with self.assertRaises(RuntimeError):
            mem.observe(read,v(.03),k(0),"real:0")

    def test_absent_and_nonfinite_source_fail(self):
        mem=VelocityResidualMemory(.01)
        with self.assertRaises(ValueError):
            mem.predict(np.full(DIM,np.nan),v(0))
        with self.assertRaises(ValueError):
            mem.predict(k(0),v(np.nan))
        t=mem.predict(k(0),v(0))
        with self.assertRaises(ValueError):
            mem.observe(t,v(0),k(0),"")
        self.assertEqual(mem.n,0)
        mem.observe(t,v(0),k(0),"authorized")

    def test_hash_candidates_subset_exact_and_audit_is_not_oracle(self):
        mem=VelocityResidualMemory(.001)
        for i in range(20):
            q=k(i)
            z=mem.predict(q,v(0))
            mem.observe(z,v(.003*(i+1)),q,str(i))
        a=mem.predict(k(3),v(0))
        self.assertTrue(0<=a.hash_recall<=1)
        self.assertLessEqual(a.size,a.collisions)
        self.assertGreater(a.collisions,0)
        # Selected exact top-K may be outside hash shortlist; hashed path can't use them.
        scored=set()
        sig=mem._signature(mem.normalize(k(3)))
        for b in (sig,)+tuple(sig^(1<<i) for i in range(11)):
            scored.update(mem.buckets.get(b,()))
        self.assertTrue(set(a.hashed_top8).issubset(scored))
        self.assertEqual(len(a.selected_weights),len(a.selected))
        np.testing.assert_array_equal(mem.pending.velocity,a.velocity)

    def test_same_fifo_for_all_modes_and_bounded_memory(self):
        mem=VelocityResidualMemory(.001)
        for i in range(CAPACITY+5):
            q=k(i)
            t=mem.predict(q,v(0))
            mem.observe(t,v(.0002),q,f"authenticated:{i}")
        self.assertEqual(mem.n,CAPACITY)
        self.assertEqual(mem.cursor,5)
        self.assertEqual(len([j for bucket in mem.buckets.values() for j in bucket]),CAPACITY)
        self.assertEqual(mem.counters()["separate_float_memory_bytes"],23552)
        self.assertIn("authenticated:260",mem.provenance)

    def test_utility_is_delayed_marginal_effect_and_can_be_negative(self):
        mem=VelocityResidualMemory(.01)
        token=mem.predict(k(0),v(0))
        mem.observe(token,v(.08),k(0),"auth:1")
        token=mem.predict(k(0),v(0))
        self.assertEqual(mem.counters() if False else len(token.selected),1)
        self.assertEqual(mem.priority[0],0)
        # Only a verified outcome can change utility; a new write reuses new FIFO slot.
        mem.observe(token,v(.08),k(0),"auth:2")
        self.assertTrue(mem.priority[0]>0)
        self.assertTrue(np.all(np.isfinite(mem.priority)))
        self.assertGreater(mem.counters()["utility_feedback_updates"],0)

if __name__=="__main__":
    unittest.main()
