"""GP-UNC-006 mathematical/software chronology falsifiers; constructed only."""
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))

from tac_osm.gp_historical_fit_diagnostic import HistoricalFITSidecar

SHA="a"*64
SRC="b"*64
BASIS={"a":([0.]*64,[1.]*64), "b":([0.]*64,[1.]*64)}

def runner(**kwargs):
    return HistoricalFITSidecar(source_sha256=SRC,checkpoint_sha256=SHA,
                                native_inducing=BASIS,noise_variance=.5,**kwargs)

def issue(row,t,entity="a",feature=0.):
    return dict(kind="issue",row_id=row,entity_id=entity,
                features=(feature,)*64,base_prediction=0.,
                feature_at=float(t),issued_at=float(t),
                checkpoint_sha256=SHA)

def label(row,t,truth):
    return dict(kind="label_arrived",row_id=row,available_at=float(t),
                truth=float(truth))

class GenuineFitContractFalsifiers(unittest.TestCase):
    def test_earlier_predictions_do_not_depend_on_later_labels(self):
        a,b=runner(),runner()
        for r in (a,b):
            r.issue(issue("first",0))
        a.label(label("first",1,2))
        b.label(label("first",1,2e6))
        later_a=a.issue(issue("later",2,"a"))
        self.assertEqual(later_a,a.issued["later"][4])
        self.assertNotEqual(later_a["native_memory_gp"],
                            b.issue(issue("later",2))["native_memory_gp"])

    def test_original_issue_has_no_truth_or_target_mask(self):
        r=runner()
        with self.assertRaises(ValueError):
            r.issue(dict(issue("x",0),truth=100))
        with self.assertRaises(ValueError):
            r.issue(dict(issue("x",0),future_observed=True))
        self.assertEqual(len(r.issued),0)

    def test_observed_outcome_cannot_change_same_time_predictions(self):
        r=runner()
        self.assertEqual(r.issue(issue("a",0))["native_memory_gp"],0.)
        r.label(label("a",1,3))
        next_p=r.issue(issue("b",1))
        self.assertEqual(next_p["native_memory_gp"],0.)
        self.assertEqual(r.updates,0)
        r.issue(issue("c",2))
        self.assertEqual(r.updates,1)

    def test_missing_original_outcome_makes_no_update(self):
        r=runner()
        r.issue(issue("a",0))
        r.missing_label("a")
        r.issue(issue("b",2))
        r.label(label("b",3,4))
        rep=r.report()
        self.assertEqual(rep["original_missing_label_receipts"],1)
        self.assertEqual(rep["original_label_events_scored"],1)
        self.assertEqual(rep["gp_real_fit_residual_update_events"],0)

    def test_two_arm_score_count_identical_for_same_truth(self):
        r=runner()
        r.issue(issue("a",0))
        r.label(label("a",1,2))
        r.issue(issue("b",2))
        r.label(label("b",3,-1))
        res=r.report()
        self.assertTrue(res["all_arm_same_outcomes"])
        self.assertEqual({x["n"] for x in res["arms"].values()},{2})
        self.assertFalse(res["model_head_to_head_superiority"])

    def test_per_entity_updates_are_separate(self):
        r=runner()
        r.issue(issue("a0",0))
        r.label(label("a0",1,5))
        b=r.issue(issue("b0",2,"b"))
        self.assertEqual(b["native_memory_gp"],0.)
        self.assertEqual(b["constant_gp"],0.)
        a=r.issue(issue("a1",3,"a"))
        self.assertGreater(a["native_memory_gp"],0.)
        self.assertEqual(r.constant["b"].observations,0)

    def test_duplicate_and_stale_clock_fails(self):
        r=runner()
        r.issue(issue("a",0))
        with self.assertRaises(ValueError):
            r.issue(issue("a",1))
        with self.assertRaises(ValueError):
            r.label(label("a",0,3))
        with self.assertRaises(ValueError):
            r.issue(issue("b",-1))
        self.assertIn("a",r.issued)

    def test_original_checkpoint_and_finite_values(self):
        r=runner()
        with self.assertRaises(ValueError):
            r.issue(dict(issue("a",0),checkpoint_sha256="c"*64))
        with self.assertRaises(ValueError):
            r.issue(dict(issue("a",0),features=(float("nan"),)*64))
        with self.assertRaises(ValueError):
            r.issue(dict(issue("a",0),issued_at=float("inf")))

    def test_source_id_and_feature_dimension_rejected(self):
        with self.assertRaises(ValueError):
            HistoricalFITSidecar(source_sha256="not-a-hash",
                                 checkpoint_sha256=SHA,native_inducing=BASIS)
        with self.assertRaises(ValueError):
            HistoricalFITSidecar(source_sha256=SRC,
                                 checkpoint_sha256=SHA,native_inducing={"a":([0.],[1.])})
        r=runner()
        with self.assertRaises(ValueError):
            r.issue(issue("a",0,"unknown"))
        with self.assertRaises(ValueError):
            r.issue(dict(issue("a",0),features=(0.,)))

    def test_empty_scoring_refuses_to_claim_result(self):
        with self.assertRaises(ValueError):
            runner().report()

    def test_strict_bounded_issued_count(self):
        r=runner(max_records=1)
        r.issue(issue("a",0))
        with self.assertRaises(ValueError):
            r.issue(issue("b",1))

if __name__=="__main__":
    unittest.main()
