"""No-fit source-graph fixtures; no real-data or capability qualification."""
import importlib.util
from pathlib import Path
import sys
import unittest

try:
    import torch
except ImportError:
    torch = None

ROOT = Path(__file__).resolve().parents[1]


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@unittest.skipIf(torch is None, "PyTorch unavailable; source graph tests NOT RUN")
class Candidate095Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        math_root = ROOT / "research" / "mathematical-engineering"
        cls.candidate = load_module("langmath095_candidate", math_root /
                                   "langmath095" / "candidate.py")
        cls.context = sys.modules["context_transition"]
        cls.upper = load_module("recurrence_only", math_root /
                                "langlayer038" / "recurrence_only.py")

    def setUp(self):
        torch.manual_seed(9501)

    def baseline(self):
        # Source-equivalent graph-free upper48/FFN455 fixture, not a checkpoint.
        model = self.context.ContextTransition026(ffn_width=455)
        model.layernorm = torch.nn.Sequential(torch.nn.LayerNorm(280),
                            self.upper.RecurrenceOnlyUpperBlock(280, 48))
        return model

    def test_default_copy_checkpoint_and_output_parity(self):
        baseline = self.baseline()
        candidate = self.candidate.TACMathCandidate095(baseline)
        self.assertEqual(baseline.parameter_count(), 499623)
        self.assertEqual(candidate.parameter_count(), baseline.parameter_count())
        self.assertEqual(set(candidate.state_dict()),
                         {"model." + key for key in baseline.state_dict()})
        for name, value in baseline.state_dict().items():
            copied = candidate.state_dict()["model." + name]
            self.assertTrue(torch.equal(value, copied), name)
            self.assertNotEqual(value.data_ptr(), copied.data_ptr(), name)
        tokens = torch.randint(0, 256, (2, 8))
        self.assertTrue(torch.equal(baseline(tokens), candidate(tokens)))
        self.assertFalse(candidate.architecture_record()["reinitialized"])

    def test_functional_copy_matches_source_forward_and_gradients(self):
        baseline = self.baseline().double()
        candidate = self.candidate.TACMathCandidate095(baseline)
        tokens = torch.randint(0, 256, (1, 7))
        mask = torch.tensor([[True, False, True, True, False, True, True]])
        actual, trace = candidate._forward(tokens, write_mask=mask,
                                          return_trace=True)
        expected, reference = baseline(tokens, write_mask=mask, return_trace=True)
        torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
        for name, value in reference.items():
            torch.testing.assert_close(trace[name], value, rtol=1e-12, atol=1e-12)
        expected_grad = torch.autograd.grad(expected.square().mean(),
                                            tuple(baseline.parameters()))
        actual_grad = torch.autograd.grad(actual.square().mean(),
                                          tuple(candidate.parameters()))
        for left, right in zip(expected_grad, actual_grad):
            torch.testing.assert_close(left, right, rtol=1e-10, atol=1e-12)

    def test_policies_preserve_causality_and_active_gradient_paths(self):
        policies = self.candidate.CandidatePolicies095(aligned_write="bounded",
                                                       positions="bounded")
        candidate = self.candidate.TACMathCandidate095(self.baseline(), policies)
        tokens = torch.randint(0, 256, (1, 8))
        poisoned = tokens.clone()
        poisoned[:, 4:] = (poisoned[:, 4:] + 17) % 256
        output = candidate(tokens)
        torch.testing.assert_close(output[:, :4], candidate(poisoned)[:, :4],
                                   rtol=1e-6, atol=1e-7)
        output.square().mean().backward()
        self.assertGreater(candidate.model.embedding.weight.grad.abs().sum(), 0)
        self.assertGreater(candidate.model.native_key.weight.grad.abs().sum(), 0)
        for parameter in candidate.parameters():
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())

    def test_bounded_write_mass_and_changing_key_trajectory(self):
        dtype = torch.float64
        state = torch.zeros(16, 8, dtype=dtype)
        for _ in range(64):
            alpha = torch.rand((), dtype=dtype)
            erase = torch.rand((), dtype=dtype)
            write = self.candidate.bounded_aligned_write095(alpha,
                                                torch.randn((), dtype=dtype))
            self.assertLessEqual(write, 1 - alpha)
            key = torch.nn.functional.normalize(torch.randn(8, dtype=dtype), dim=0)
            value = torch.randn(16, dtype=dtype).tanh()
            state = state @ (alpha * (torch.eye(8, dtype=dtype) -
                            erase * torch.outer(key, key))) + write * torch.outer(value, key)
            self.assertLessEqual(float(state.norm()), 4 + 1e-12)
        candidate = self.candidate.TACMathCandidate095(self.baseline(),
                    self.candidate.CandidatePolicies095(aligned_write="bounded"))
        _, trace = candidate(torch.randint(0, 256, (1, 9)), return_trace=True)
        self.assertTrue((trace["aligned_write"] <= 1 - trace["aligned_alpha"]).all())

    def test_position_policy_and_graph_lesion_accounting(self):
        indices = torch.tensor([0, 1, 1024, 10**12])
        changed = self.candidate.bounded_position_features095(indices, torch.float64)
        original = self.context.phi_position(indices, torch.float64)
        self.assertTrue(((changed[:, 0] >= 0) & (changed[:, 0] < 1)).all())
        self.assertTrue(torch.equal(changed[:, 1:], original[:, 1:]))
        candidate = self.candidate.TACMathCandidate095(self.baseline(),
                      self.candidate.CandidatePolicies095(concept_graph="disabled"))
        output, trace = candidate(torch.randint(0, 256, (1, 6)), return_trace=True)
        self.assertEqual(torch.count_nonzero(trace["concept_laplacian"]), 0)
        torch.testing.assert_close(trace["concept_posterior"],
                    (trace["concept_evidence"] + .25 * trace["concept_prewrite"]) / 1.25)
        record = candidate.architecture_record()
        self.assertEqual(record["inactive_graph_parameters"], 512)
        self.assertEqual(record["active_parameters"], candidate.parameter_count() - 512)
        output.square().mean().backward()
        self.assertIsNone(candidate.model.concept_q.weight.grad)
        self.assertIsNone(candidate.model.concept_k.weight.grad)

    def test_multiscale_is_explicit_reinitialization_of_copy(self):
        baseline = self.baseline()
        old_state = {key: value.clone() for key, value in baseline.state_dict().items()}
        with self.assertRaises(ValueError):
            self.candidate.TACMathCandidate095(baseline,
                    self.candidate.CandidatePolicies095(upper_retention="multiscale"))
        candidate = self.candidate.make_multiscale_candidate095(baseline)
        block = candidate.model.layernorm[1]
        self.assertEqual(torch.count_nonzero(block.forget.weight), 0)
        horizons = -1 / block.forget.bias.sigmoid().log()
        torch.testing.assert_close(horizons[[0, -1]],
                    torch.tensor([2., 4096.]), rtol=3e-4, atol=1e-4)
        self.assertTrue(candidate.architecture_record()["reinitialized"])
        self.assertEqual(candidate.parameter_count(), baseline.parameter_count())
        for key, value in baseline.state_dict().items():
            self.assertTrue(torch.equal(value, old_state[key]), key)
        tokens = torch.randint(0, 256, (1, 7))
        output = candidate(tokens)
        self.assertEqual(output.shape, (1, 7, 256))
        output.square().mean().backward()
        self.assertTrue(torch.isfinite(block.forget.bias.grad).all())

    def test_10m_graph_accepts_policies_and_multiscale(self):
        source = load_module("langmath095_source10m", ROOT / "research" /
                             "language-scaling" / "lang10m" / "model.py")
        baseline = source.TACLanguage10M001()
        candidate = self.candidate.make_multiscale_candidate095(baseline,
                    self.candidate.CandidatePolicies095(aligned_write="bounded"))
        self.assertEqual(candidate.parameter_count(), 10001188)
        self.assertEqual(candidate.architecture_record()["retention_initialization"]["upper_blocks"], 4)
        output = candidate(torch.randint(0, 4096, (1, 3)))
        self.assertEqual(output.shape, (1, 3, 4096))
        self.assertTrue(torch.isfinite(output).all())


if __name__ == "__main__":
    unittest.main()
