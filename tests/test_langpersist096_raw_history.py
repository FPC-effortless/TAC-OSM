import importlib.util
from pathlib import Path
import unittest
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('langpersist096_raw_history',ROOT/'research/mathematical-engineering/langpersist096/raw_history.py')
raw=importlib.util.module_from_spec(spec); spec.loader.exec_module(raw)


class RawHistoryTests(unittest.TestCase):
    def test_distribution_has_no_future_interface_and_normalizes(self):
        observed=b'keyA keyA keyA key'
        p=raw.distribution(observed)
        self.assertAlmostEqual(float(p.sum()),1.)
        self.assertTrue((p>0).all())
        self.assertEqual(int(p.argmax()),ord('A'))
        self.assertTrue(np.array_equal(p,raw.distribution(observed)))

    def test_generation_is_causal_bounded_and_input_pure(self):
        a=np.frombuffer((b'abcd '*26)[:128],dtype=np.uint8)[None].copy()
        q=np.frombuffer((b'abcd '*13)[:64],dtype=np.uint8)[None].copy()
        before_a=a.copy(); before_q=q.copy()
        out=raw.predict(a,q)
        self.assertEqual(out['log_probs'].shape,(1,1,256))
        self.assertEqual(out['greedy'].shape,(1,1,4))
        self.assertEqual(out['greedy'][0,0,0],out['log_probs'][0,0].argmax())
        self.assertEqual(out['metadata']['costs']['state_payload_bytes'],3904)
        np.testing.assert_array_equal(a,before_a); np.testing.assert_array_equal(q,before_q)


if __name__=='__main__': unittest.main()
