"""Metric/cohort falsifiers use constructed inputs, never DEV model scores."""
import math
import unittest
import torch

from common import Totals, batch_bytes, coordinates, identity, load_plan
from evaluate import score_logits
from aggregate import bootstrap_difference


class EvalChecks(unittest.TestCase):
    def test_no_cross_file_or_incomplete_tail_and_unique_targets(self):
        rows = [{"path":"b", "bytes":257, "split":"DEV"}, {"path":"a", "bytes":256, "split":"DEV"},
                {"path":"fit", "bytes":900, "split":"FIT"}]
        windows = coordinates(rows, "DEV")
        self.assertEqual(windows, [("a", 0), ("b", 0), ("b", 128)])
        targets = [(n, t) for n, o in windows for t in range(o+1, o+129)]
        self.assertEqual(len(targets), len(set(targets)))

    def test_withheld_target_does_not_enter_input(self):
        raw = bytes(range(129)); changed = raw[:-1] + bytes([201])
        x, y = batch_bytes({"f":raw}, [("f", 0)])
        xx, yy = batch_bytes({"f":changed}, [("f", 0)])
        self.assertEqual(x, xx); self.assertNotEqual(y, yy)
        with self.assertRaises(ValueError): batch_bytes({"f":b"short"}, [("f", 0)])

    def test_uniform_logits_exact_nll(self):
        logits = torch.zeros(2, 3, 256); y = torch.tensor([[1, 2, 3], [4, 5, 6]])
        loss, correct = score_logits(logits, y, torch)
        self.assertTrue(torch.allclose(loss, torch.full_like(loss, math.log(256)), atol=1e-12, rtol=0))
        self.assertEqual(correct.sum().item(), 0)

    def test_wrong_target_falsifier_and_nonfinite_rejection(self):
        logits = torch.zeros(1, 1, 256); logits[..., 7] = 10
        good, _ = score_logits(logits, torch.tensor([[7]]), torch)
        bad, _ = score_logits(logits, torch.tensor([[8]]), torch)
        self.assertAlmostEqual((bad-good).item(), 10, places=10)
        logits[..., 3] = float("nan")
        with self.assertRaises(ValueError): score_logits(logits, torch.tensor([[7]]), torch)

    def test_weighted_group_denominators(self):
        r = Totals([{"path":"a", "group":"g"}, {"path":"b", "group":"g"}])
        r.add("a", 4, 1, 2); r.add("b", 6, 2, 6)
        v = r.finish()
        self.assertEqual(v["targets"], 8); self.assertEqual(v["nll"], 1.25)
        self.assertEqual(v["by_group"]["g"]["targets"], 8)
        self.assertEqual(v["accuracy"], 3/8)

    def test_frozen_protocol_has_all_original_models(self):
        p = load_plan()
        self.assertEqual(len(p["checkpoint_locks"]), 18)
        self.assertEqual(p["evaluation"]["dev_targets"], 594688)
        self.assertNotEqual(identity([("a", 0)]), identity([("a", 128)]))

    def test_paired_bootstrap_preserves_constant_difference(self):
        def metric(value):
            return {"nll":value, "by_group":{str(i):{"targets":i+1, "nll_sum":value*(i+1)} for i in range(17)}}
        out = bootstrap_difference([metric(2)]*3, [metric(1)]*3,
            {"rng_seed":10501, "replicates":50, "interval_percentiles":[2.5, 97.5]})
        self.assertEqual(out["mean_paired_nll_difference"], 1)
        self.assertEqual(out["descriptive_95_percentile_interval"], [1, 1])
        wrong = metric(1); wrong["by_group"]["0"]["targets"] = 90
        with self.assertRaises(ValueError):
            bootstrap_difference([metric(2)]*3, [wrong]*3,
                {"rng_seed":10501, "replicates":50, "interval_percentiles":[2.5, 97.5]})


if __name__ == "__main__": unittest.main()
