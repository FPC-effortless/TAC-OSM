"""GP-UNC-003 constructed offline replay boundary, no eligible real-source input."""
from dataclasses import replace
from pathlib import Path
import sys
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tac_osm.gp_prequential_nofit import RecordedForecast, replay_constructed_only


def case(i, *, entity="a", issue=0., observed=1., available=2., target=1.):
    return RecordedForecast(
        row_id=str(i), entity_id=entity, features=(0.,),
        base_prediction=0., truth=target, feature_at=issue,
        issued_at=issue, outcome_observed_at=observed,
        outcome_available_at=available)


class PrequentialNoFitContract(unittest.TestCase):
    def test_delayed_labels_not_read_before_available(self):
        rows = [case("a", issue=0, observed=1, available=4, target=7.),
                case("b", issue=2, observed=3, available=6),
                case("c", issue=5, observed=6, available=8)]
        output = replay_constructed_only(rows, {"a": [[0.]]})
        self.assertEqual([x.eligible_past_updates for x in output], [0,0,1])
        self.assertEqual(output[0].corrected_prediction, 0.)
        self.assertEqual(output[1].corrected_prediction, 0.)
        self.assertGreater(output[2].corrected_prediction, 0.)

    def test_future_truth_poison_does_not_change_preoutcome_prediction(self):
        rows = [case("a", issue=0, observed=1, available=4, target=3.),
                case("b", issue=2, observed=3, available=6)]
        a = replay_constructed_only(rows, {"a":[[0.]]})
        b = replay_constructed_only([replace(rows[0], truth=1_000_000.), rows[1]],
                                    {"a":[[0.]]})
        self.assertEqual(a, b)

    def test_distinct_entities_do_not_cross_condition(self):
        rows = [case("a1", issue=0, observed=1, available=2, target=7.),
                case("b1", entity="b", issue=3, observed=4, available=5),
                case("a2", issue=6, observed=7, available=8)]
        out = replay_constructed_only(rows, {"a":[[0.]], "b":[[0.]]})
        self.assertEqual(out[1].eligible_past_updates, 0)
        self.assertEqual(out[1].corrected_prediction, 0.)
        self.assertEqual(out[2].eligible_past_updates, 1)
        self.assertGreater(out[2].corrected_prediction, 0.)

    def test_uncertainty_only_updates_after_eligible_feedback(self):
        rows = [case("a", issue=0, observed=1, available=4, target=3.),
                case("b", issue=2, observed=3, available=6),
                case("c", issue=5, observed=6, available=8)]
        out = replay_constructed_only(rows, {"a":[[0.]]})
        self.assertAlmostEqual(out[0].gp_latent_variance,
                               out[1].gp_latent_variance)
        self.assertLess(out[2].gp_latent_variance, out[1].gp_latent_variance)

    def test_duplicate_unsorted_future_and_missing_entity_rejected(self):
        ok = case("a", issue=0, observed=1, available=2)
        invalid = [
            [ok, ok],
            [replace(ok, row_id="b", issued_at=4, feature_at=4,
                     outcome_observed_at=5, outcome_available_at=6), ok],
            [replace(ok, outcome_observed_at=0)],
            [replace(ok, entity_id="unknown")],
        ]
        for rows in invalid:
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                replay_constructed_only(rows, {"a":[[0.]]})

    def test_pending_queue_has_explicit_external_memory_cap(self):
        rows = [case(str(i), issue=i, observed=100+i, available=110+i)
                for i in range(4)]
        with self.assertRaises(ValueError):
            replay_constructed_only(rows, {"a":[[0.]]}, max_pending=2)
        self.assertEqual(len(replay_constructed_only(rows, {"a":[[0.]]},
                                                           max_pending=4)), 4)

    def test_nonfinite_predictions_and_truth_rejected(self):
        for key in ("base_prediction", "truth", "feature_at"):
            row = replace(case("a"), **{key: float("nan")})
            with self.subTest(key=key), self.assertRaises(ValueError):
                replay_constructed_only([row], {"a":[[0.]]})


if __name__ == "__main__":
    unittest.main()
