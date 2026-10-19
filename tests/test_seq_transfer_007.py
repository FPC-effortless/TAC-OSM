"""Validate teacher/no-teacher experiments without accessing TEST."""
import unittest
import torch
from scripts.run_seq_transfer_007 import PILOTS, CONDITIONS, data, train
from scripts.run_seq_transfer_006 import ARMS


class TestTeacherAblation(unittest.TestCase):
    def test_registered_matrix(self):
        self.assertEqual(len(PILOTS), 2)
        self.assertEqual(set(CONDITIONS), {"final_only", "dense_teacher"})
        self.assertEqual(CONDITIONS["final_only"], 0.0)
        self.assertEqual(CONDITIONS["dense_teacher"], 0.25)
        self.assertEqual(set(ARMS), {"tac_gated", "routed_gru", "global_gru"})

    def test_inputs_reused(self):
        tr, val, p = data()
        self.assertEqual(len(tr[0]), 3000)
        self.assertEqual(len(val["L128_E4"][0]), 600)
        for k in ("train", "L64_E4", "L128_E4"):
            self.assertEqual(len(p[k]["sha256_public_inputs_and_labels"]), 64)

    def test_both_losses_update_weights_without_test(self):
        tr, val, p = data()
        mini = tuple(x[:4] for x in tr)
        small = {"L64_E4": tuple(x[:4] for x in val["L64_E4"])}
        for condition, weight in CONDITIONS.items():
            out = train("routed_gru", condition, weight, PILOTS[0], mini,
                        small, 1, __import__("pathlib").Path("unused"),
                        persist=False)
            self.assertEqual(out["updates"], 1)
            self.assertTrue(all(0<=score<=1 for score in out["validation"]["L64_E4"].values()))

if __name__ == "__main__":
    unittest.main()
