"""Algebraic falsifiers, not trained intelligence evidence."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'research/mathematical-engineering/langmath095'))
import audit


class AuditTests(unittest.TestCase):
    def test_known_retention_and_aligned_gain_counterexample(self):
        retention = audit.retention_report()
        self.assertAlmostEqual(retention['half_life_updates'],138.282572986,places=7)
        gain = audit.aligned_gain_report()
        self.assertGreater(gain['constant_effective_key']['equilibrium_gain'],2500)
        self.assertLessEqual(gain['constant_effective_key']['bounded_policy_equilibrium_gain'],1)
        self.assertEqual(gain['status'],'PASS')

    def test_address_load_separates_fit_capacity_from_whole_model_claim(self):
        rows = audit.memory_load_report()['rows']
        self.assertLess(rows[3]['best_linear_mse'],1e-20)
        self.assertGreater(rows[-1]['best_linear_mse'],.1)
        self.assertEqual(rows[-1]['address_rank'],8)

    def test_uniform_pool_cancels_graph_and_nonuniform_pool_exposes_it(self):
        result = audit.concept_report()
        self.assertEqual(result['status'],'PASS')
        self.assertLess(result['uniform_pool_graph_effect_max_abs'],1e-12)
        self.assertGreater(result['sensitivity'][-1]['edge_directional_derivative_l2'],1e-5)
        self.assertLessEqual(result['condition_number'],2.2)

    def test_position_fixture_loses_lexical_signal(self):
        rows = audit.position_report()['rows']
        self.assertLess(rows[-1]['illustrative_lexical_difference'],1e-5)
        self.assertGreater(rows[-1]['bounded_position_lexical_difference'],.1)

    def test_source_pins_and_absent_checkpoint_remain_distinct(self):
        report = audit.build_report()
        self.assertTrue(all(x['status']=='PASS' for x in report['source_inventory']))
        self.assertEqual(report['status'],'PASS_CONSTRUCTED_CHECKS')
        self.assertEqual(report['checkpoint_diagnostics']['status'],'INCONCLUSIVE')
        self.assertEqual(report['optimizer_steps'],0)
        json.dumps(report,allow_nan=False)
        with tempfile.TemporaryDirectory() as tmp:
            self.assertTrue(all(x['status']=='INCONCLUSIVE' for x in audit.source_inventory(tmp)))

    def test_output_is_append_only_and_partial_checkpoint_args_rejected(self):
        with self.assertRaises(ValueError):
            audit.build_report(checkpoint=Path('not-read'))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'result.json'
            self.assertEqual(audit.main(['--output',str(path)]),0)
            with self.assertRaises(FileExistsError):
                audit.main(['--output',str(path)])

    def test_checkpoint_failure_and_missing_source_cannot_report_pass(self):
        import types
        with patch.dict(sys.modules,{'checkpoints':types.SimpleNamespace(audit_checkpoint=lambda *a,**k:{'status':'FAIL'})}):
            result = audit.build_report(checkpoint='c',inputs='i',receipt='r')
        self.assertEqual(result['status'],'FAIL')
        with patch.dict(sys.modules,{'checkpoints':types.SimpleNamespace(audit_checkpoint=lambda *a,**k:{'status':'BLOCKED_EXECUTION'})}):
            self.assertEqual(audit.build_report(checkpoint='c',inputs='i',receipt='r')['status'],'FAIL')
        with patch.object(audit,'source_inventory',return_value=[{'status':'INCONCLUSIVE'}]):
            self.assertEqual(audit.build_report()['status'],'INCONCLUSIVE')


if __name__ == '__main__':
    unittest.main()
