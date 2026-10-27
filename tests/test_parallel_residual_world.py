"""NO-FIT checks for proposed parallel-state residual FFN.

All inputs are fabricated *numeric unit fixtures* only. These tests cannot
be cited as real-world skill, forecast accuracy, or fair-compute evidence.
"""
import unittest

import torch

from tac_osm.variants.parallel_residual_world import ParallelResidualWorld


class ParallelResidualWorldNoFitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(271027)
        self.model = ParallelResidualWorld()
        self.observation = torch.randn(2, 5, 14) * 0.1
        self.action = torch.randn(2, 5, 14) * 0.1

    def test_live_parameter_budget_and_no_extra_recurrent_state(self):
        active = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.assertEqual(active, 96303)
        self.assertTrue(95000 <= active <= 105000)
        state = self.model.init_stream(2)
        self.assertEqual(state.fast_sum.shape, (2, 88))
        self.assertEqual(state.memory.shape, (2, 8, 8))
        self.assertEqual((state.fast_sum.numel() + state.memory.numel()) // 2, 152)
        self.assertEqual(state.count, 0)

    def test_parallel_batch_matches_streamed_outputs_and_memory(self):
        self.model.eval()
        batch, trace = self.model(
            self.observation, self.action, return_trace=True,
        )
        state = self.model.init_stream(2)
        streamed = []
        for t in range(self.observation.shape[1]):
            y, state, step_trace = self.model.stream_step(
                state, self.observation[:, t], self.action[:, t],
                return_trace=True,
            )
            streamed.append(y)
            torch.testing.assert_close(
                step_trace["post_scan_correction"],
                trace["post_scan_correction"][:, t],
                atol=2e-4, rtol=5e-4,
            )
        torch.testing.assert_close(
            torch.stack(streamed, dim=1), batch,
            atol=2e-4, rtol=5e-4,
        )
        torch.testing.assert_close(
            state.memory, trace["memory"][:, -1],
            atol=2e-4, rtol=5e-4,
        )
        self.assertEqual(state.count, self.observation.shape[1])

    def test_all_ffn_parameters_are_live_and_differentiable(self):
        prediction = self.model(self.observation, self.action)
        loss = (prediction - self.observation).square().mean()
        loss.backward()
        for name, param in self.model.named_parameters():
            if name.startswith("residual_ffn") or name == "residual_logit":
                self.assertIsNotNone(param.grad, name)
                self.assertTrue(torch.isfinite(param.grad).all(), name)
                self.assertGreater(param.grad.abs().sum().item(), 0.0, name)

    def test_ffn_is_postscan_and_bounded_by_parent_scale(self):
        result, trace = self.model(
            self.observation, self.action, return_trace=True,
        )
        base, original_trace = self.model.core(
            self.observation, self.action, return_trace=True,
        )
        torch.testing.assert_close(
            trace["memory"], original_trace["memory"], atol=0, rtol=0,
        )
        torch.testing.assert_close(
            result - base,
            self.model.core.delta_scale * trace["post_scan_correction"],
            atol=1e-7, rtol=1e-5,
        )
        self.assertLessEqual(
            trace["post_scan_correction"].abs().max().item(),
            trace["post_scan_gate"].item() + 1e-7,
        )
        with torch.no_grad():
            self.model.residual_logit.fill_(-100.0)
        zeroed = self.model(self.observation, self.action)
        torch.testing.assert_close(zeroed, base, atol=1e-7, rtol=0)

    def test_future_observation_cannot_affect_past_predictions(self):
        before = self.model(self.observation, self.action)
        altered_obs = self.observation.clone()
        altered_action = self.action.clone()
        altered_obs[:, 3:] += 50
        altered_action[:, 3:] -= 75
        after = self.model(altered_obs, altered_action)
        torch.testing.assert_close(before[:, :3], after[:, :3], atol=1e-6, rtol=0)

    def test_only_six_family_factory_can_build_untrained_residual(self):
        from tac_osm.variants.parallel_six import (
            instantiate_untrained_postscan_core,
        )
        for family in ("physical", "temporal", "system1", "enterprise",
                       "edge_lite", "counterfactual"):
            model = instantiate_untrained_postscan_core(
                family, state_dim=14, action_dim=14,
            )
            self.assertIsInstance(model, ParallelResidualWorld)
            self.assertEqual(
                sum(p.numel() for p in model.parameters() if p.requires_grad),
                96303,
            )
        with self.assertRaises(ValueError):
            instantiate_untrained_postscan_core(
                "ced_seventh_family", state_dim=14, action_dim=14,
            )

    def test_input_validation_and_stream_input_nonmutation(self):
        with self.assertRaises(ValueError):
            self.model(torch.randn(2, 14), self.action)
        state = self.model.init_stream(2)
        original_sum = state.fast_sum.clone()
        original_memory = state.memory.clone()
        _, next_state = self.model.stream_step(
            state, self.observation[:, 0], self.action[:, 0],
        )
        torch.testing.assert_close(state.fast_sum, original_sum, atol=0, rtol=0)
        torch.testing.assert_close(state.memory, original_memory, atol=0, rtol=0)
        self.assertIsNot(next_state, state)
        with self.assertRaises(ValueError):
            self.model.stream_step(state, self.observation[:, 0], torch.zeros(2, 2))


if __name__ == "__main__":
    unittest.main()
