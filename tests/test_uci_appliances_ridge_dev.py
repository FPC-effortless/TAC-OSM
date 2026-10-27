"""Constructed ridge fit/replay checks, not real source or performance evidence."""
import importlib.util
import json
import math
from pathlib import Path
import sys
import unittest

ROOT=Path(__file__).resolve().parents[1]/"specialist_models"
sys.path.insert(0,str(ROOT))
spec=importlib.util.spec_from_file_location("uci_appliances_ridge_dev",ROOT/"uci_appliances_ridge_dev.py")
m=importlib.util.module_from_spec(spec)
sys.modules[spec.name]=m
spec.loader.exec_module(m)


class RidgeFixtureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.values=[90+20*(i%144<30)+4*math.sin(2*math.pi*i/144)+i%9
                    for i in range(m.source.DEV_END)]

    def test_constructed_fit_and_dev_are_explicitly_not_science(self):
        model=m.fit(self.values)
        r=m.evaluate(model,self.values)
        self.assertEqual(model["fit_examples"],m.source.FIT_END-m.source.H-144)
        self.assertEqual(r["scientific_promotion"],"NOT_QUALIFIED")
        self.assertEqual(r["dev"]["n"],m.source.DEV_END-m.source.CAL_END-m.source.H)
        self.assertFalse(r["independent_test_scored"])
        self.assertTrue(0<=r["dev"]["nominal_90_coverage"]<=1)
        self.assertEqual(model["model_type"],"classical_fit_only_ridge")

    def test_post_fit_data_cannot_change_original_weights(self):
        original=m.fit(self.values)
        manipulated=self.values.copy()
        manipulated[m.source.FIT_END+50]+=999999
        manipulated[m.source.CAL_END+1]+=999999
        self.assertEqual(original["coefficients"],m.fit(manipulated)["coefficients"])
        self.assertEqual(original["normalizer_fit_means"],m.fit(manipulated)["normalizer_fit_means"])

    def test_checkpoint_json_replays_identical_predictions(self):
        model=m.fit(self.values)
        reloaded=json.loads(json.dumps(model))
        pred=m.forecast(model,self.values,m.source.CAL_END+10)
        self.assertEqual(pred,m.forecast(reloaded,self.values,m.source.CAL_END+10))

    def test_never_uses_future_inputs_at_inference(self):
        model=m.fit(self.values)
        altered=self.values.copy()
        i=m.source.CAL_END+5
        altered[i+m.source.H]=10000000.
        self.assertEqual(m.forecast(model,self.values,i),m.forecast(model,altered,i))

    def test_no_protected_target_evaluation(self):
        model=m.fit(self.values)
        r=m.evaluate(model,self.values)
        self.assertFalse(r["protected_test_targets_parsed"])
        self.assertEqual(r["source_zip_sha256"],m.PINNED_UCI374_SHA)

    def test_bad_checkpoint_is_blocked(self):
        model=m.fit(self.values)
        model["source_zip_sha256"]="0"*64
        with self.assertRaises(m.source.SourceError):
            m.forecast(model,self.values,m.source.CAL_END+15)

    def test_exact_linear_solver(self):
        w=m.solve([[2,0],[0,4]],[6,8])
        self.assertEqual(w,[3,2])

    def test_bad_lag_index_blocked(self):
        with self.assertRaises(m.source.SourceError):
            m.feature(self.values,0)

if __name__=="__main__":
    unittest.main()
