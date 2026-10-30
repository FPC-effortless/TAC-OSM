"""No-fit falsifiers for acquisition gradients, masking and target alignment."""
import importlib.util
from pathlib import Path
import sys
import unittest

import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / "research/mathematical-engineering/langacquire097"


def load(name):
    spec = importlib.util.spec_from_file_location("langacquire097_objective_test_" + name,
                                                HERE / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


models, objective = load("models"), load("objective")


class Objective097Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_exact_first_and_later_target_alignment(self):
        x, y, query = objective.episode_tensors(b"abc", b"de", b"WXYZ")
        self.assertEqual(bytes(x[0].tolist()), b"abcdeWXY")
        self.assertEqual(bytes(y[0].tolist()), b"bcdeWXYZ")
        self.assertEqual(query[0].tolist(), [False] * 4 + [True] * 4)

    def test_query_loss_reaches_acquisition_state_and_acquisition_only_tokens(self):
        for arm in models.ARMS:
            with self.subTest(arm=arm):
                model = models.make_model(arm, 9701)
                # Bytes0..7 occur only in acquisition, never in later inputs.
                a = bytes(range(8)); q = bytes(range(40, 46)); y = bytes(range(80, 84))
                x, target, mask = objective.episode_tensors(a, q, y)
                result = objective.query_loss(model, x, target, mask,
                                               acquisition_end=len(a), retain_acquisition_grad=True)
                result.loss.backward()
                receipt = objective.acquisition_gradient_receipt(result)
                self.assertEqual(receipt["status"], "PASS")
                self.assertGreater(float(model.core.embedding.weight.grad[:8].abs().sum()), 0.)
                self.assertEqual(result.supervised_targets, len(y))
                self.assertEqual(result.observed_inputs, len(a) + len(q) + len(y) - 1)

    def test_masked_target_poison_does_not_change_query_loss_or_gradients(self):
        x, y, mask = objective.episode_tensors(b"abcdefgh", b"ijkl", b"MNOP")
        altered = y.clone(); altered[~mask] = 255 - altered[~mask]
        for arm in models.ARMS:
            original, changed = (models.make_model(arm, 9701) for _ in range(2))
            a = objective.query_loss(original, x, y, mask)
            b = objective.query_loss(changed, x, altered, mask)
            torch.testing.assert_close(a.loss, b.loss, atol=0., rtol=0.)
            a.loss.backward(); b.loss.backward()
            for (_, pa), (_, pb) in zip(original.named_parameters(), changed.named_parameters()):
                torch.testing.assert_close(pa.grad, pb.grad, atol=0., rtol=0.)

    def test_full_and_query_use_same_logits_with_distinct_supervision(self):
        x, target, mask = objective.episode_tensors(b"abcdefgh", b"ijkl", b"MNOP")
        for arm in models.ARMS:
            model = models.make_model(arm, 9701)
            q = objective.query_loss(model, x, target, mask)
            full = objective.full_loss(model, x, target, mask)
            torch.testing.assert_close(q.logits, full.logits, atol=0., rtol=0.)
            ce = F.cross_entropy(full.logits.flatten(0, 1), target.flatten(), reduction="none")
            torch.testing.assert_close(full.loss, ce.mean())
            torch.testing.assert_close(q.loss, ce[mask.flatten()].mean())
            self.assertEqual(q.supervised_targets, 4)
            self.assertEqual(full.supervised_targets, x.numel())

    def test_right_padding_cannot_add_supervision_or_change_real_logits(self):
        x, y, mask = objective.episode_tensors(b"abc", b"de", b"WXYZ")
        padded_x = F.pad(x, (0, 4), value=0)
        padded_y = F.pad(y, (0, 4), value=0)
        padded_q = F.pad(mask, (0, 4), value=False)
        valid = F.pad(torch.ones_like(mask), (0, 4), value=False)
        for arm in models.ARMS:
            model = models.make_model(arm, 9701)
            original = objective.query_loss(model, x, y, mask)
            padded = objective.query_loss(model, padded_x, padded_y, padded_q, valid_mask=valid)
            torch.testing.assert_close(original.loss, padded.loss, atol=3e-6, rtol=1e-5)
            torch.testing.assert_close(original.logits, padded.logits[:, :x.shape[1]], atol=3e-6, rtol=1e-5)
            self.assertEqual(original.supervised_targets, padded.supervised_targets)
        invalid = valid.clone(); invalid[:, -1] = True
        with self.assertRaises(ValueError):
            objective.query_loss(model, padded_x, padded_y, padded_q, valid_mask=invalid)

    def test_fails_closed_for_missing_query_targets_or_wrong_masks(self):
        model = models.make_model("gru", 9701)
        x = torch.zeros(2, 8, dtype=torch.long)
        target = torch.ones_like(x); mask = torch.zeros_like(x, dtype=torch.bool)
        with self.assertRaises(ValueError):
            objective.query_loss(model, x, target, mask)
        mask[:, -2:] = True
        with self.assertRaises(ValueError):
            objective.query_loss(model, x, target, mask.float())
        with self.assertRaises(ValueError):
            objective.query_loss(model, x, target, mask, retain_acquisition_grad=True)
        with self.assertRaises(ValueError):
            objective.sequence_loss(model, x, target, mask, objective="unregistered")


if __name__ == "__main__":
    unittest.main()
