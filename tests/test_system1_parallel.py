"""Engineering/invariant tests only; constructed arrays are NOT skill evidence."""
import unittest
import torch

from tac_osm.variants.system1_parallel import System1ParallelWorld


class System1ParallelTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(28016)
        self.model = System1ParallelWorld().double().eval()

    def inputs(self, batch=2, length=9):
        return (torch.randn(batch, length, 14, dtype=torch.float64) * 0.2,
                torch.randn(batch, length, 14, dtype=torch.float64) * 0.1)

    def test_six_family_shared_backbone_and_parameter_preflight(self):
        self.assertEqual(sum(p.numel() for p in self.model.parameters()), 95907)
        self.assertEqual(self.model.carried_fp32_bytes, 608)
        x, a = self.inputs()
        y = self.model(x, a)
        self.assertEqual(tuple(y.next_observation.shape), (2, 9, 14))
        self.assertEqual(tuple(y.adverse_event_logit.shape), (2, 9))
        self.assertEqual(tuple(y.residual_scale.shape), (2, 9, 14))
        self.assertTrue(torch.all(y.residual_scale > 0).item())
        self.assertTrue(torch.all((y.adverse_event_score > 0) &
                                  (y.adverse_event_score < 1)).item())
        self.assertFalse(y.is_calibrated)
        self.assertFalse(y.authorized_to_act)
        self.assertEqual(y.scope, "advisory_only")

    def test_stream_and_parallel_prefix_agree_with_same_weights(self):
        for steps in (1, 3, 9, 16):
            x, a = self.inputs(length=steps)
            with torch.no_grad():
                batch = self.model(x, a)
                state = self.model.initial_state(2)
                streamed = []
                for t in range(steps):
                    y, state = self.model.step(state, x[:, t], a[:, t])
                    streamed.append(y)
                self.assertEqual(state.count, steps)
                self.assertEqual(state.fast_sum.numel() + state.memory.numel(),
                                 2 * 152)
                for field in ("next_observation", "adverse_event_logit",
                              "adverse_event_score", "residual_scale"):
                    actual = torch.stack([getattr(v, field) for v in streamed], dim=1)
                    torch.testing.assert_close(actual, getattr(batch, field),
                                               atol=2e-7, rtol=2e-7)
                self.assertLessEqual(torch.linalg.matrix_norm(
                    state.memory, ord="fro").max().item(), 4.000001)

    def test_future_changes_do_not_modify_past_predictions(self):
        x, a = self.inputs(length=11)
        with torch.no_grad():
            original = self.model(x, a)
            xx = x.clone(); aa = a.clone()
            xx[:, 6:] += 1000
            aa[:, 6:] -= 1000
            changed = self.model(xx, aa)
            for field in ("next_observation", "adverse_event_logit",
                          "adverse_event_score", "residual_scale"):
                torch.testing.assert_close(getattr(original, field)[:, :6],
                                           getattr(changed, field)[:, :6],
                                           atol=2e-8, rtol=2e-8)

    def test_end_to_end_autograd_reaches_shared_memory_and_advisory_head(self):
        x, a = self.inputs(length=7)
        result = self.model(x.requires_grad_(), a)
        labels = torch.randint(0, 2, (2, 7), dtype=torch.float64)
        labels_next = torch.randn_like(result.next_observation) * 0.1 + x
        risk_loss = torch.nn.functional.binary_cross_entropy_with_logits(
            result.adverse_event_logit, labels)
        normal_loss = (
            (labels_next - result.next_observation).square() /
            (2 * result.residual_scale.square()) +
            result.residual_scale.log()
        ).mean()
        (risk_loss + normal_loss).backward()
        for p in (self.model.core.encoder.weight,
                  self.model.core.memory_gates.weight,
                  self.model.reactive_head[0].weight,
                  self.model.reactive_head[2].weight):
            self.assertIsNotNone(p.grad)
            self.assertTrue(torch.isfinite(p.grad).all().item())
            self.assertGreater(p.grad.abs().sum().item(), 0)
        self.assertTrue(torch.isfinite(x.grad).all().item())

    def test_batch_and_stream_loss_gradients_agree_fp64(self):
        import copy
        x, a = self.inputs(length=5)
        batch_model = copy.deepcopy(self.model)
        step_model = copy.deepcopy(self.model)

        z = batch_model(x, a)
        loss_batch = (0.003 * z.next_observation.square().sum() +
                      z.adverse_event_logit.square().mean() +
                      z.residual_scale.square().mean())
        loss_batch.backward()

        state = step_model.initial_state(2)
        seq = []
        for t in range(5):
            y, state = step_model.step(state, x[:, t], a[:, t])
            seq.append(y)
        y_observed = torch.stack([u.next_observation for u in seq], 1)
        y_logit = torch.stack([u.adverse_event_logit for u in seq], 1)
        y_scale = torch.stack([u.residual_scale for u in seq], 1)
        loss_stream = (0.003 * y_observed.square().sum() +
                       y_logit.square().mean() +
                       y_scale.square().mean())
        loss_stream.backward()

        torch.testing.assert_close(loss_batch, loss_stream, atol=1e-8, rtol=1e-8)
        for (name, p), (name2, q) in zip(batch_model.named_parameters(),
                                         step_model.named_parameters()):
            self.assertEqual(name, name2)
            self.assertIsNotNone(p.grad, name)
            self.assertIsNotNone(q.grad, name)
            torch.testing.assert_close(p.grad, q.grad, atol=1e-7, rtol=1e-6)

    def test_fail_closed_invalid_or_mismatched_inputs(self):
        x, a = self.inputs()
        bad = x.clone(); bad[0, 0, 0] = float("nan")
        with self.assertRaises(ValueError):
            self.model(bad, a)
        with self.assertRaises(ValueError):
            self.model(x, a[..., :13])
        with self.assertRaises(ValueError):
            self.model(x[:, :0], a[:, :0])
        with self.assertRaises(ValueError):
            self.model.step(self.model.initial_state(1), x[:, 0], a[:, 0])
        with self.assertRaises(ValueError):
            System1ParallelWorld(head_width=0)
        with self.assertRaises(ValueError):
            self.model.initial_state(True)


if __name__ == "__main__":
    unittest.main()
