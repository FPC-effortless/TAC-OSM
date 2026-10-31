"""SENSE-TOK-002 unfitted neural graph checks; NEVER step an optimizer."""
import unittest

import torch
from tac_osm.causal_byte_probe import (
    BOS_ID, FEATURE_DIM, CausalByteGRU, LearnedSensoryGateGRU,
    features_from_view, quick_features, parameter_receipt, predecision_examples,
)
from tac_osm.causal_patch_stream import CausalPatchStream, STREAM_VARIANTS


class TestCausalByteProbeNoFit(unittest.TestCase):
    def test_actual_parameter_count_and_matching(self):
        control = parameter_receipt(CausalByteGRU())
        gate = parameter_receipt(LearnedSensoryGateGRU())
        self.assertEqual(control["trainable_parameters"], 100400)
        self.assertEqual(gate["trainable_parameters"], 100513)
        self.assertEqual(gate["deployed_learned_parameters"], 100513)
        self.assertTrue(control["absolute_100k_band"] and gate["absolute_100k_band"])
        self.assertLessEqual(abs(100513 - 100400) / 100400, 0.01)
        self.assertFalse(control["training_authorized"])
        self.assertFalse(gate["training_authorized"])

    def test_reference_features_match_fast_prefix_features(self):
        data = b"abc123 !!!def987\x00\xff\n" * 3
        for v in STREAM_VARIANTS:
            stream = CausalPatchStream(variant=v)
            for b in data:
                self.assertEqual(quick_features(stream), features_from_view(stream.snapshot()))
                stream.observe(bytes([b]))
            self.assertEqual(quick_features(stream), features_from_view(stream.snapshot()))

    def test_next_byte_inputs_precede_targets(self):
        for v in STREAM_VARIANTS:
            ids, feats, targets = predecision_examples(b"abc123", v)
            self.assertEqual(ids.tolist(), [BOS_ID] + list(b"abc12"))
            self.assertEqual(targets.tolist(), list(b"abc123"))
            self.assertEqual(feats.shape, (6, FEATURE_DIM))
            self.assertEqual(feats[0, :8].tolist(), [1.0] + [0.0] * 7)
            self.assertTrue(bool(torch.isfinite(feats).all()))

    def test_gru_future_suffix_poison_noninterference(self):
        torch.manual_seed(129)
        model = CausalByteGRU().eval()
        prefix = b"source code prefix(x):\n"
        with torch.no_grad():
            for v in STREAM_VARIANTS:
                xa, fa, _ = predecision_examples(prefix + b"AAAA not observed", v)
                xb, fb, _ = predecision_examples(prefix + b"9999 withheld    ", v)
                k = len(prefix)
                self.assertTrue(torch.equal(xa[:k+1], xb[:k+1]))
                self.assertTrue(torch.equal(fa[:k+1], fb[:k+1]))
                self.assertTrue(torch.equal(model(xa[None], fa[None])[:, :k+1],
                                            model(xb[None], fb[None])[:, :k+1]))

    def test_gru_source_to_output_backward_without_fit(self):
        torch.manual_seed(771)
        model = CausalByteGRU().train()
        raw = b"function test(): return 12345\n" * 2
        for v in STREAM_VARIANTS:
            model.zero_grad(set_to_none=True)
            ids, features, target = predecision_examples(raw, v)
            logits = model(ids[None], features[None])
            self.assertEqual(logits.shape, (1, len(raw), 256))
            torch.nn.functional.cross_entropy(logits[0], target).backward()
            for name, p in model.named_parameters():
                self.assertIsNotNone(p.grad, name)
                self.assertTrue(bool(torch.isfinite(p.grad).all()), name)
                self.assertGreater(float(p.grad.abs().sum()), 0.0, name)
        # No optimizer constructed/stepped.

    def test_learned_gate_gradient_and_future_poison(self):
        torch.manual_seed(82)
        model = LearnedSensoryGateGRU().eval()
        raw = b"hello 123 data code\n" * 2
        ids, _, target = predecision_examples(raw, "fixed")
        logits, gates = model(ids[None], return_gates=True)
        self.assertEqual(logits.shape, (1, len(raw), 256))
        self.assertEqual(gates.shape, (1, len(raw), 1))
        torch.nn.functional.cross_entropy(logits[0], target).backward()
        self.assertGreater(float(model.boundary_gate.weight.grad.abs().sum()), 0.0)
        for name, p in model.named_parameters():
            self.assertIsNotNone(p.grad, name)
            self.assertTrue(bool(torch.isfinite(p.grad).all()), name)
        with torch.no_grad():
            x1 = predecision_examples(b"abc123defAAAA", "fixed")[0]
            x2 = predecision_examples(b"abc123def9999", "fixed")[0]
            out1, g1 = model(x1[None], return_gates=True)
            out2, g2 = model(x2[None], return_gates=True)
            self.assertTrue(torch.equal(out1[:, :10], out2[:, :10]))
            self.assertTrue(torch.equal(g1[:, :10], g2[:, :10]))

    def test_empty_and_invalid_shapes(self):
        ids, features, targets = predecision_examples(b"", "fixed")
        self.assertEqual(ids.numel(), 0)
        self.assertEqual(features.shape, (0, 16))
        self.assertEqual(targets.numel(), 0)
        with self.assertRaises(ValueError):
            CausalByteGRU()(torch.zeros((1, 2), dtype=torch.long),
                            torch.zeros((1, 2, 15)))
