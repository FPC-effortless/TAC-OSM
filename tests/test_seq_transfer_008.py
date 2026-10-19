"""SEQ008 preflight: source honesty, update causality, and sealed TEST."""
import unittest

import numpy as np
import torch

from scripts.run_seq_transfer_008 import (
    ARMS, MAX_ENTITIES, PILOT_SEEDS, RESERVED_TEST_SEEDS, TRAIN_SEED,
    VALIDATION_SEEDS, _sample, fit_final_only, generate, replay,
)


class TestSEQ008(unittest.TestCase):
    def test_test_sources_inaccessible(self):
        for seed in RESERVED_TEST_SEEDS:
            for purpose in ("train", "validation", "test"):
                with self.assertRaises(PermissionError):
                    generate(seed, 36, (64,), 4, purpose=purpose)
        with self.assertRaises(PermissionError):
            generate(VALIDATION_SEEDS[0], 36, (64,), 4, purpose="train")

    def test_repeatable_balanced_sources_and_masks(self):
        a, manifest = generate(TRAIN_SEED, 36, (32, 64), 4, purpose="train")
        b, other = generate(TRAIN_SEED, 36, (32, 64), 4, purpose="train")
        self.assertEqual(manifest["sha256_inputs_labels"], other["sha256_inputs_labels"])
        self.assertTrue(all(torch.equal(x, y) for x, y in zip(a, b)))
        ids, ops, active, truth, q, task, labels = a
        self.assertEqual(ids.shape, (36, 64))
        self.assertEqual(truth.shape, (36, 64, MAX_ENTITIES))
        self.assertEqual(len(set(q[:, 0].tolist())), 4)
        for row in manifest["task_labels"].values():
            self.assertEqual(row, {"n": 12, "positive": 6})
        for j in range(36):
            end = int(active[j].sum().item())
            states = replay(ids[j, :end].numpy(), ops[j, :end].numpy(), end)
            self.assertTrue(np.array_equal(truth[j, :end].numpy(), states))
            aa, bb = q[j].tolist()
            expected = int(states[-1, aa])
            if int(task[j]) == 2:
                expected ^= int(states[-1, bb])
            self.assertEqual(int(labels[j].item()), expected)
            self.assertTrue(torch.equal(truth[j, end - 1], truth[j, -1]))

    def test_randomized_update_positions_and_causal_last_write(self):
        rng = np.random.default_rng(992822)
        positions = set()
        for kind in range(3):
            for _ in range(36):
                ids, ops, states, qa, qb, label, meta = _sample(rng, 128, 4, kind)
                targets = {qa, qb}
                pos = [i for i, e in enumerate(ids) if e in targets]
                last = pos[-1]
                positions.add(last)
                self.assertLess(last, meta["cutoff"])
                self.assertGreater(meta["last_target_lag"], 25)
                self.assertTrue(0 <= label <= 1)
                self.assertTrue(np.array_equal(states, replay(ids, ops, len(ids))))
                modified = ops.copy()
                old_op = int(modified[last])
                modified[last] = (1 - old_op) if old_op in (0, 1) else 3
                changed = replay(ids, modified, len(ids))
                different = int(changed[-1, qa])
                if kind == 2:
                    different ^= int(changed[-1, qb])
                self.assertNotEqual(label, different, "Last relevant update was not causal")
                if kind == 2:
                    for target in (qa, qb):
                        sequence = [int(ops[i]) for i in pos if int(ids[i]) == target]
                        self.assertIn(sequence[0], (0, 1))
                        self.assertTrue(all(op == 2 for op in sequence[1:]))
                    oddness = [
                        (sum(int(ops[i]) == 2 for i in pos if int(ids[i]) == target) % 2)
                        for target in (qa, qb)
                    ]
                    self.assertNotEqual(*oddness, "XOR toggles cancel across both entities")
        self.assertGreater(len(positions), 15, "Important events are at fixed positions")

    def test_final_only_trains_with_no_intermediate_teacher(self):
        batch, _ = generate(TRAIN_SEED, 36, (32,), 4, purpose="train")
        val, _ = generate(VALIDATION_SEEDS[0], 36, (32,), 4, purpose="validation")
        for arm in ARMS:
            result = fit_final_only(
                arm, PILOT_SEEDS[0], batch, {"v": val}, 1,
                __import__("pathlib").Path("unused"), save=False
            )
            self.assertEqual(result["train_condition"], "final_label_only")
            self.assertEqual(result["steps"], 1)
            self.assertTrue(all(0 <= v <= 1 for v in result["validation"]["v"].values()))


if __name__ == "__main__":
    unittest.main()
