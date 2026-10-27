"""GP-UNC-004 constructed event-stream information-flow falsifiers."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tac_osm.gp_event_boundary_nofit import replay_truth_isolated_constructed as replay

CHECKPOINT = "a" * 64
Z = {"a": [[0.0]], "b": [[0.0]]}


def issue(i, t, *, entity="a", x=(0.0,), base=0.0):
    return {"kind": "issue", "row_id": i, "entity_id": entity, "features": x,
            "base_prediction": base, "feature_at": t, "issued_at": t,
            "checkpoint_sha256": CHECKPOINT}


def label(i, t, truth):
    return {"kind": "label_arrived", "row_id": i,
            "available_at": t, "truth": truth}


def run(events, **kwargs):
    return replay(events, Z, checkpoint_sha256=CHECKPOINT, **kwargs)


class TruthIsolatedGP(unittest.TestCase):
    def test_future_truth_not_in_issue_contract(self):
        record = issue("a0", 0)
        record["truth"] = 42
        with self.assertRaisesRegex(ValueError, "exclude truth"):
            run([record])
        record.pop("truth")
        record["outcome_available_at"] = 10
        with self.assertRaisesRegex(ValueError, "exclude truth"):
            run([record])

    def test_default_frozen_receives_no_adaptive_updates(self):
        out = run([issue("a0", 0), label("a0", 1, 5),
                   issue("a1", 2)])
        self.assertEqual(out[1].corrected_prediction, 0)
        self.assertEqual(out[1].eligible_past_updates, 0)

    def test_explicit_adaptive_label_only_after_arrival(self):
        out = run([issue("a0", 0), issue("a1", .5),
                   label("a0", 1, 4), issue("a2", 2)],
                  adaptive_feedback=True)
        self.assertEqual(out[0].corrected_prediction, 0)
        self.assertEqual(out[1].corrected_prediction, 0)
        self.assertEqual(out[2].eligible_past_updates, 1)
        self.assertGreater(out[2].corrected_prediction, 0)

    def test_equal_time_arrival_not_admitted_until_later_clock(self):
        out = run([issue("a0", 0), label("a0", 1, 3),
                   issue("a1", 1), issue("a2", 2)],
                  adaptive_feedback=True)
        self.assertEqual(out[1].corrected_prediction, 0)
        self.assertEqual(out[1].eligible_past_updates, 0)
        self.assertGreater(out[2].corrected_prediction, 0)

    def test_future_label_mutation_cannot_change_earlier_outputs(self):
        prefix = [issue("a0", 0), issue("a1", .25)]
        one = run(prefix + [label("a0", 1, 4), issue("a2", 2)],
                  adaptive_feedback=True)
        two = run(prefix + [label("a0", 1, 4e8), issue("a2", 2)],
                  adaptive_feedback=True)
        self.assertEqual(one[:2], two[:2])
        self.assertNotEqual(one[2].corrected_prediction, two[2].corrected_prediction)

    def test_per_entity_updates_are_isolated(self):
        out = run([issue("a0", 0), label("a0", 1, 5),
                   issue("b0", 2, entity="b"),
                   issue("a1", 3)], adaptive_feedback=True)
        self.assertEqual(out[1].eligible_past_updates, 0)
        self.assertEqual(out[1].corrected_prediction, 0)
        self.assertEqual(out[2].eligible_past_updates, 1)

    def test_duplicate_ids_and_orphan_labels_rejected(self):
        for events in ([label("no-forecast", 0, 2)],
                       [issue("a0", 0), issue("a0", 1)],
                       [issue("a0", 0), label("a0", 1, 2),
                        label("a0", 2, 3)]):
            with self.subTest(events=events), self.assertRaises(ValueError):
                run(events, adaptive_feedback=True)

    def test_reversed_event_times_and_future_features_rejected(self):
        for events in ([issue("a0", 2), issue("a1", 1)],
                       [issue("a0", 0), label("a0", 0, 1)],
                       [{**issue("a0", 0), "feature_at": 1}],
                       [{**issue("a0", 0), "issued_at": float("nan")}]):
            with self.subTest(events=events), self.assertRaises(ValueError):
                run(events)

    def test_checkpoint_mismatch_rejected(self):
        bad = {**issue("a0", 0), "checkpoint_sha256": "b" * 64}
        with self.assertRaisesRegex(ValueError, "checkpoint"):
            run([bad])

    def test_bounded_audit_memory_and_excess_entities(self):
        with self.assertRaises(ValueError):
            run([issue("a0", 0), issue("a1", 1)], max_pending=1)
        with self.assertRaises(ValueError):
            run([issue("a0", 0), issue("a1", 1)], max_records=1)
        with self.assertRaises(ValueError):
            run([issue("a0", 0)], max_entities=1)

    def test_generator_does_not_preconsume_later_labels(self):
        def events():
            yield issue("a0", 0)
            # Executed only when the runtime requests the next event.
            yield issue("a1", 1)
            yield label("a0", 2, 7)
            yield issue("a2", 3)
        out = run(events(), adaptive_feedback=True)
        self.assertEqual([x.eligible_past_updates for x in out], [0, 0, 1])


if __name__ == "__main__":
    unittest.main()
