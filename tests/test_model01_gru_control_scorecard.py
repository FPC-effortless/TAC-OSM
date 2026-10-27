"""Source-backed original Git model manifests for reproducible comparisons."""
import copy,json,sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"specialist_models"))
import model01_gru_control_scorecard as m

class OriginalControlEvidence(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tac=json.loads(m.TAC.read_bytes())
        cls.gru=json.loads(m.GRU.read_bytes())

    def test_original_manifest_grounded_metrics(self):
        r=m.evaluate(self.tac,self.gru)
        self.assertEqual(len(r["native_seed_summaries"]),3)
        self.assertLess(r["relative_parameter_gap"],.01)
        self.assertFalse(r["independent_qualification"])
        self.assertFalse(r["neural_architecture_superiority_claim"])
        self.assertFalse(r["training_FLOPs_or_tuning_budget_equal"])
        self.assertEqual(r["native_seed_summaries"][0]["real_DEV_forecast_count_per_model"],1968)

    def test_model_skill_metric_tradeoff_reported(self):
        r=m.evaluate(self.tac,self.gru)
        self.assertLess(r["mean_GRU_MAE_relative_to_TAC"],0.)
        self.assertGreater(r["mean_GRU_interval_score_relative_to_TAC"],0.)

    def test_fake_extra_training_updates_rejected(self):
        t=copy.deepcopy(self.gru);t["training_steps"]+=1
        with self.assertRaises(m.ScorecardError):m.evaluate(self.tac,t)

    def test_source_changed_rejected(self):
        t=copy.deepcopy(self.gru);t["source_zip_sha256"]="0"*64
        with self.assertRaises(m.ScorecardError):m.evaluate(self.tac,t)

    def test_sampler_broken_rejected(self):
        t=copy.deepcopy(self.gru);t["runs"][0]["sample_order_sha256"]="bad"
        with self.assertRaises(m.ScorecardError):m.evaluate(self.tac,t)

    def test_original_gru_weights_replaced_rejected(self):
        t=copy.deepcopy(self.gru);t["runs"][0]["checkpoint_sha256"]="0"*64
        with self.assertRaises(m.ScorecardError):m.evaluate(self.tac,t)

    def test_claim_of_strict_compute_parity_rejected(self):
        t=copy.deepcopy(self.gru);t["training_FLOPs_matched"]=True
        with self.assertRaises(m.ScorecardError):m.evaluate(self.tac,t)

    def test_fake_independent_source_result_rejected(self):
        t=copy.deepcopy(self.gru);t["independent_evaluation"]=True
        with self.assertRaises(m.ScorecardError):m.evaluate(self.tac,t)

if __name__=="__main__":
    unittest.main()
