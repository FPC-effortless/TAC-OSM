"""No-fit mathematical fixtures. These do not measure learned capability."""
import math
import random
import unittest

from research.langcache104.math_reference import bank_step, causal_attention, rotate_pairs, rotary_step, softmax


class ReferenceTests(unittest.TestCase):
    def test_rank_write_preserves_cone(self):
        rng = random.Random(104)
        for rank in (1, 2, 5):
            memory, mass = [[0.]*4 for _ in range(6)], [0.]*4
            for _ in range(100):
                a = rng.random()
                w = (1-a)*rng.random()
                keys = [softmax([rng.uniform(-4, 4) for _ in range(4)]) for _ in range(rank)]
                values = [[math.tanh(rng.uniform(-4, 4)) for _ in range(6)] for _ in range(rank)]
                q = softmax([rng.uniform(-4, 4) for _ in range(4)])
                reads, memory, mass, evidence = bank_step(q, keys, values, a, w, memory, mass)
                self.assertTrue(all(0 <= z <= 1+1e-12 for z in mass))
                self.assertTrue(all(abs(memory[i][j]) <= mass[j]+1e-12
                                    for i in range(6) for j in range(4)))
                self.assertTrue(all(abs(x) <= 1+1e-12 for x in reads))
                self.assertGreaterEqual(evidence, 0)

    def test_freeze_and_erase_are_distinct(self):
        memory, mass = [[.5, -.25]], [.5, .25]
        _, frozen_m, frozen_z, _ = bank_step([.5, .5], [[.4, .6]], [[.7]],
                                            1., 0., memory, mass)
        self.assertEqual(frozen_m, memory)
        self.assertEqual(frozen_z, mass)
        _, erased_m, erased_z, _ = bank_step([.5, .5], [[.4, .6]], [[.7]],
                                            0., 0., memory, mass)
        self.assertEqual(erased_m, [[0., 0.]])
        self.assertEqual(erased_z, [0., 0.])

    def test_rotations_preserve_norm_and_convex_offer_envelope(self):
        x = [.4, -.6, .2, .8]
        theta = [.8, -1.3]
        y = rotate_pairs(x, theta)
        norm = lambda z: math.sqrt(sum(v*v for v in z))
        self.assertAlmostEqual(norm(x), norm(y), places=14)
        offer = [-.5, .4, .9, -.2]
        out = rotary_step(.7, theta, offer, x)
        self.assertLessEqual(norm(out), .7*norm(x)+.3*norm(offer)+1e-14)
        self.assertEqual(rotary_step(1., [0., 0.], offer, x), x)

    def test_exact_recent_attention_uses_only_supplied_resident_window(self):
        # Positional/address selection is a neural operation; softmax itself
        # is still a mixture, not an exact byte-record lookup.
        out = causal_attention([0., 0.], [[1., 2.], [3., 4.]], [[99.], [7.]], 1)
        self.assertEqual(out, [7.])
        out = causal_attention([0., 0.], [[1., 2.], [3., 4.]], [[99.], [7.]], 2)
        self.assertEqual(out, [53.])

    def test_stable_softmax_large_logits(self):
        self.assertEqual(softmax([10000., 10000.]), [.5, .5])


if __name__ == "__main__":
    unittest.main()
