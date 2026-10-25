"""Unit-only synthetic numeric fixtures. Never used as capability evidence."""
import unittest
import numpy as np
from tac_osm.variants.outcome_utility_memory import OnlineMemory,CAPACITY

def key(i):
    z=np.zeros(8,dtype=np.float32)
    z[i%8]=1
    return z

def value(v):
    return np.ones(14,dtype=np.float32)*v

class TestDelayedUtilityMemory(unittest.TestCase):
    def test_feedback_requires_prediction_first(self):
        m=OnlineMemory("utility",capacity=8)
        t=m.predict(key(0),value(0))
        with self.assertRaises(RuntimeError):
            m.predict(key(1),value(0))
        with self.assertRaises(RuntimeError):
            m.observe(None,value(1),key(0),"source-A-t1")
        self.assertEqual(len([x for x in m.slots if x is not None]),0)
        m.observe(t,value(.02),key(0),"source-A-t1")
        self.assertEqual(m.stats()["occupied"],1)
        with self.assertRaises(RuntimeError):
            m.observe(t,value(.02),key(0),"source-A-t1")
        self.assertEqual(m.stats()["utility_updates"],0)

    def test_utility_only_after_real_next_outcome(self):
        m=OnlineMemory("utility",capacity=8)
        z=m.predict(key(0),value(0))
        m.observe(z,value(.04),key(0),"src")
        t=m.predict(key(0),value(0))
        self.assertTrue(t.hit)
        before=m.slots[0].utility
        self.assertEqual(before,0)
        # positive future truth after candidate-prediction improvement.
        m.observe(t,value(.04),key(0),"src")
        self.assertGreater(m.slots[0].utility,0)
        self.assertEqual(m.stats()["utility_updates"],1)

    def test_neutral_same_budget_and_no_utility(self):
        n=OnlineMemory("neutral",capacity=2)
        u=OnlineMemory("utility",capacity=2)
        for step in range(3):
            for bank in (n,u):
                token=bank.predict(key(step),value(0))
                bank.observe(token,value(.03),key(step),f"source-{step}")
        self.assertEqual(n.stats()["occupied"],2)
        self.assertEqual(u.stats()["occupied"],2)
        self.assertEqual(n.stats()["float_payload_bytes"],u.stats()["float_payload_bytes"])
        self.assertEqual(n.stats()["utility_updates"],0)
        self.assertTrue(all(x.utility==0 for x in n.slots))

    def test_valid_sources_and_finite_inputs(self):
        m=OnlineMemory("neutral",capacity=2)
        with self.assertRaises(ValueError):
            m.predict(np.ones(7),value(0))
        with self.assertRaises(ValueError):
            m.predict(key(1),value(np.nan))
        t=m.predict(key(1),value(0))
        with self.assertRaises(ValueError):
            m.observe(t,value(1),key(1),"")
        self.assertEqual(m.stats()["occupied"],0)
        m.observe(t,value(.1),key(1),"ok")

    def test_hash_bucket_membership_after_replacement(self):
        m=OnlineMemory("neutral",capacity=1)
        for i in range(4):
            tok=m.predict(key(i),value(0))
            m.observe(tok,value(.001),key(i),f"observed-{i}")
            self.assertEqual(sum(len(v) for v in m.buckets.values()),1)
            self.assertEqual(m.stats()["occupied"],1)
        self.assertEqual(m.stats()["replacements"],3)

    def test_query_determinism_and_recall_audit_cannot_change_predictions(self):
        a,b=OnlineMemory("neutral",capacity=8),OnlineMemory("neutral",capacity=8)
        for i in range(8):
            for bank in (a,b):
                tok=bank.predict(key(i),value(0))
                bank.observe(tok,value(.01*(i+1)),key(i),f"src-{i}")
        qa=a.predict(key(3),value(0),audit=True)
        qb=b.predict(key(3),value(0),audit=False)
        np.testing.assert_array_equal(qa.forecast,qb.forecast)
        self.assertIsNotNone(qa.recall)
        self.assertGreaterEqual(qa.recall,0)
        self.assertLessEqual(qa.recall,1)

if __name__=="__main__":
    unittest.main()
