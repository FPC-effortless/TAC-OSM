"""GRAPH016 constructed operator tests. Never treat as real-data predictive skill."""
import math
import unittest
import numpy as np
from graph016_online_expert_mix import CausalExpertMixture

class OnlineMathTests(unittest.TestCase):
    def test_pathwise_telescope_and_log_k_regret(self):
        rng=np.random.default_rng(1601)
        for T in (1,7,128,4096):
            m=CausalExpertMixture(('a','b'))
            for _ in range(T):
                experts=rng.dirichlet(np.ones(7),size=(2,3))
                law=m.predict(experts)
                self.assertTrue(np.allclose(law.sum(-1),1))
                m.observe(rng.integers(7,size=2),('a','b'))
            r=m.receipt()
            best=np.array(r['expert_log_loss'])
            expected=-np.logaddexp.reduce(-best,axis=1)+math.log(3)
            np.testing.assert_allclose(r['mixture_log_loss'],expected,rtol=1e-12,atol=5e-11)
            self.assertTrue(all(v <= math.log(3)+1e-8 for v in r['fixed_expert_regret']))

    def test_chooses_good_expert_from_prior_outcomes_only(self):
        m=CausalExpertMixture(('one',))
        e=np.array([[[.95,.05],[.2,.8],[.5,.5]]])
        for _ in range(30):
            m.predict(e)
            m.observe(np.array([0]),('one',))
        self.assertGreater(m.receipt()['final_weights'][0][0],.999999)

    def test_causal_prefix_unchanged_by_future(self):
        p=np.array([[[.7,.3],[.5,.5],[.4,.6]]])
        a=CausalExpertMixture(('a',));b=CausalExpertMixture(('a',))
        np.testing.assert_array_equal(a.predict(p),b.predict(p))
        a.observe(np.array([1]),('a',));b.observe(np.array([1]),('a',))
        q=np.array([[[.2,.8],[.9,.1],[.3,.7]]])
        np.testing.assert_array_equal(a.predict(q),b.predict(q))

    def test_mismatched_owner_and_order_fails_closed(self):
        m=CausalExpertMixture(('a',))
        p=np.ones((1,3,3))/3
        with self.assertRaises(ValueError): m.observe(np.array([0]),('a',))
        m.predict(p)
        with self.assertRaises(ValueError):m.predict(p)
        with self.assertRaises(ValueError):m.observe(np.array([0]),('another',))
        m.observe(np.array([0]),('a',))

    def test_owner_states_do_not_cross(self):
        m=CausalExpertMixture(('a','b'))
        p=np.array([[[.99,.01],[.01,.99],[.5,.5]],
                    [[.99,.01],[.01,.99],[.5,.5]]])
        for _ in range(40):
            m.predict(p)
            m.observe(np.array([0,1]),('a','b'))
        w=m.receipt()['final_weights']
        self.assertGreater(w[0][0],.99)
        self.assertGreater(w[1][1],.99)

    def test_reject_invalid_probabilities_and_owners(self):
        m=CausalExpertMixture(('a',))
        p=np.ones((1,3,2))/2
        for bad in (p*2,p*0,np.full_like(p,np.nan)):
            with self.assertRaises(ValueError):m.predict(bad)
        with self.assertRaises(ValueError):CausalExpertMixture(('a','a'))

    def test_identical_experts_exactly_preserve_law(self):
        m=CausalExpertMixture(('a',))
        p=np.array([[[.4,.6],[.4,.6],[.4,.6]]])
        for _ in range(31):
            np.testing.assert_allclose(m.predict(p)[0],p[0,0])
            m.observe(np.array([1]),('a',))
        self.assertAlmostEqual(m.receipt()['fixed_expert_regret'][0],0,places=10)

if __name__ == '__main__':
    unittest.main()
