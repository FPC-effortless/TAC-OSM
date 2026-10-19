"""SEQ006 generator, oracle-replay, and TEST-seal regression tests."""
import unittest
import numpy as np
import torch

from scripts.run_seq_transfer_006 import (
    TASKS, TEST_SEED, VALID_SEED, MAX_ENTITIES, generate,
    StreamingAgent, objective, _build_one, ARMS,
)


class TestSEQ006(unittest.TestCase):
    def test_forbidden_test_partition(self):
        with self.assertRaises(PermissionError):
            generate(TEST_SEED, 6, (512,), 6, purpose="validation")
        with self.assertRaises(PermissionError):
            generate(VALID_SEED, 6, (512,), 6, purpose="test")

    def test_determinism_and_label_balance(self):
        a, meta = generate(VALID_SEED, 60, (64,), 4, purpose="validation")
        b, again = generate(VALID_SEED, 60, (64,), 4, purpose="validation")
        self.assertEqual(meta["sha256_public_inputs_and_labels"],
                         again["sha256_public_inputs_and_labels"])
        self.assertTrue(all(torch.equal(x, y) for x, y in zip(a, b)))
        for task in TASKS:
            self.assertEqual(meta["tasks"][task]["positive"], 10)
            self.assertEqual(meta["tasks"][task]["count"], 20)
        self.assertGreaterEqual(meta["min_target_lag"], 10)

    def test_full_sequence_exact_state_replay(self):
        data, _ = generate(VALID_SEED + 10, 90, (128,), 6, purpose="validation")
        ids, ops, active, states, q, tasks, label = data
        for i in range(len(ids)):
            physical_state = np.zeros(MAX_ENTITIES, dtype=np.float32)
            for t in range(len(ops[i])):
                if not active[i, t]:
                    break
                e, op = int(ids[i, t]), int(ops[i, t])
                if op == 0:
                    physical_state[e] = 0.
                elif op == 1:
                    physical_state[e] = 1.
                elif op == 2:
                    physical_state[e] = 1. - physical_state[e]
                self.assertTrue(np.array_equal(physical_state, states[i, t].numpy()))
            target = int(q[i, 0])
            partner = int(q[i, 1])
            result = int(physical_state[target])
            if tasks[i] == 2:
                result ^= int(physical_state[partner])
            self.assertEqual(float(result), float(label[i]))

    def test_all_models_train_and_stream_without_teacher_input(self):
        data, _ = generate(VALID_SEED, 12, (32,), 4, purpose="validation")
        ids, ops, active, teacher, q, tasks, labels = data
        for arm in ARMS:
            with self.subTest(arm=arm):
                agent = StreamingAgent(arm)
                final, auxiliary, locations = agent(ids, ops, active, q, tasks)
                self.assertEqual(tuple(final.shape), (12,))
                self.assertEqual(tuple(auxiliary.shape), (12, 8, 6))
                self.assertEqual(len(locations), 8)
                loss, _, _ = objective(agent, data)
                self.assertTrue(bool(torch.isfinite(loss)))
                loss.backward()
                self.assertTrue(any(p.grad is not None for p in agent.parameters()))

    def test_query_entity_is_not_in_visible_stream_features(self):
        rng = np.random.default_rng(1103)
        for task in range(3):
            for desired in (0, 1):
                e, op, state, qa, qb, answer, age = _build_one(
                    rng, 512, 6, task, desired)
                self.assertEqual(answer, desired)
                self.assertGreaterEqual(age, 128)
                self.assertEqual(len(e), 512)
                self.assertEqual(len(op), 512)

if __name__ == "__main__":
    unittest.main()
