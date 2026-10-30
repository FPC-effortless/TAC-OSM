"""Constructed no-fit source/parity/causality checks, not capability evidence."""
import importlib.util
from pathlib import Path
import sys
import unittest

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "research/mathematical-engineering/langacquire097/models.py"
SPEC = importlib.util.spec_from_file_location("langacquire097_models_test", PATH)
models = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = models
SPEC.loader.exec_module(models)


class Models097Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_live_counts_all_active_and_pairwise_within_one_percent(self):
        measured = []
        tokens = torch.arange(40).reshape(2, 20)
        for arm in models.ARMS:
            with self.subTest(arm=arm):
                model = models.make_model(arm, 9701)
                counts = model.counts()
                self.assertEqual(counts["trainable"], models.PARAMETERS[arm])
                self.assertEqual(counts["deployed"], counts["trainable"])
                self.assertTrue(95000 <= counts["trainable"] <= 105000)
                logits, _ = model(tokens)
                logits[:, -4:].square().mean().backward()
                for name, parameter in model.named_parameters():
                    self.assertIsNotNone(parameter.grad, (arm, name))
                    self.assertTrue(torch.isfinite(parameter.grad).all(), (arm, name))
                    self.assertGreater(float(parameter.grad.abs().sum()), 0., (arm, name))
                measured.append(counts["trainable"])
        self.assertLessEqual((max(measured) - min(measured)) / min(measured), .01)

    def test_source_wrappers_match_unchanged_recurrent_sources(self):
        tokens = torch.arange(24).reshape(2, 12)
        owners = ("source-a", "source-b")
        for arm in ("tac", "gru", "ssm"):
            with self.subTest(arm=arm):
                model = models.make_model(arm, 9701).eval()
                original = type(model.core)()
                original.load_state_dict(model.core.state_dict(), strict=True)
                expected, oldstate = original(tokens, owners=owners)
                actual, state = model(tokens, owners=owners)
                torch.testing.assert_close(actual, expected, atol=0., rtol=0.)
                expected_tensors = oldstate.tensors if arm == "tac" else {
                    "hidden": oldstate.value.transpose(0, 1) if arm == "gru" else oldstate.value}
                for name, value in state.tensors.items():
                    torch.testing.assert_close(value, expected_tensors[name], atol=0., rtol=0.)

    def test_transformer_matches_standard_pytorch_encoder(self):
        model = models.make_model("transformer", 9701).eval().core
        tokens = torch.arange(26).reshape(2, 13)
        x = model.embedding(tokens)
        position = torch.arange(13, dtype=x.dtype)[:, None]
        frequency = torch.pow(torch.tensor(10000., dtype=x.dtype),
                              -torch.arange(0, 48, 2, dtype=x.dtype) / 48)
        pe = torch.zeros(13, 48)
        pe[:, 0::2] = torch.sin(position * frequency)
        pe[:, 1::2] = torch.cos(position * frequency)
        x = x + pe
        forbidden = torch.triu(torch.ones(13, 13, dtype=torch.bool), diagonal=1)
        for block in model.blocks:
            reference = nn.TransformerEncoderLayer(48, 4, 290, dropout=0.,
                                                    activation="gelu", batch_first=True,
                                                    norm_first=True).eval()
            reference.self_attn.in_proj_weight.data.copy_(block.attention.qkv.weight)
            reference.self_attn.in_proj_bias.data.copy_(block.attention.qkv.bias)
            reference.self_attn.out_proj.load_state_dict(block.attention.projection.state_dict())
            reference.norm1.load_state_dict(block.norm1.state_dict())
            reference.norm2.load_state_dict(block.norm2.state_dict())
            reference.linear1.load_state_dict(block.up.state_dict())
            reference.linear2.load_state_dict(block.down.state_dict())
            x = reference(x, src_mask=forbidden)
        expected = model.readout(model.norm(x))
        actual, _ = model(tokens)
        torch.testing.assert_close(actual, expected, atol=3e-6, rtol=1e-5)

    def test_chunking_preserves_logits_state_and_gradients(self):
        tokens = torch.arange(30).reshape(2, 15)
        owners = ("a", "b")
        for arm in models.ARMS:
            with self.subTest(arm=arm):
                whole = models.make_model(arm, 9701)
                split = models.make_model(arm, 9701)
                a, final = whole(tokens, owners=owners)
                first, early = split(tokens[:, :7], owners=owners)
                snapshot = {k: v.detach().clone() for k, v in early.tensors.items()}
                last, streamed = split(tokens[:, 7:], state=early, owners=owners)
                b = torch.cat((first, last), dim=1)
                torch.testing.assert_close(a, b, atol=3e-5, rtol=1e-4)
                self.assertEqual(final.clock, streamed.clock)
                for name, value in final.tensors.items():
                    torch.testing.assert_close(value, streamed.tensors[name], atol=3e-5, rtol=1e-4)
                    torch.testing.assert_close(early.tensors[name], snapshot[name], atol=0., rtol=0.)
                a[:, -3:].square().mean().backward()
                b[:, -3:].square().mean().backward()
                for (_, pa), (_, pb) in zip(whole.named_parameters(), split.named_parameters()):
                    torch.testing.assert_close(pa.grad, pb.grad, atol=3e-5, rtol=1e-4)

    def test_future_poison_and_entity_isolation(self):
        tokens = torch.arange(28).reshape(2, 14)
        poisoned = tokens.clone(); poisoned[:, 9:] = 255 - poisoned[:, 9:]
        owners = ("a", "b")
        for arm in models.ARMS:
            with self.subTest(arm=arm):
                model = models.make_model(arm, 9701).eval()
                expected, state = model(tokens, owners=owners)
                changed, _ = model(poisoned, owners=owners)
                torch.testing.assert_close(expected[:, :9], changed[:, :9], atol=3e-5, rtol=1e-4)
                for i in range(2):
                    one, own = model(tokens[i:i+1], owners=(owners[i],))
                    torch.testing.assert_close(expected[i:i+1], one, atol=3e-5, rtol=1e-4)
                    for name in state.tensors:
                        torch.testing.assert_close(state.tensors[name][i:i+1], own.tensors[name],
                                                   atol=3e-5, rtol=1e-4)

    def test_complete_interventions_preserve_clock_metadata_and_original(self):
        owners = ("a", "b")
        for arm in models.ARMS:
            model = models.make_model(arm, 9701)
            _, state = model(torch.arange(20).reshape(2, 10), owners=owners)
            before = {name: tensor.detach().clone() for name, tensor in state.tensors.items()}
            for mode in ("carry", "reset", "donor", "corrupt"):
                altered = model.intervene(state, mode, [1, 0] if mode == "donor" else None)
                self.assertEqual(altered.owners, state.owners)
                self.assertEqual(altered.clock, state.clock)
                for name, tensor in altered.tensors.items():
                    torch.testing.assert_close(state.tensors[name], before[name], atol=0., rtol=0.)
                    if mode == "reset":
                        self.assertEqual(float(tensor.abs().sum()), 0.)
                    elif mode == "donor":
                        torch.testing.assert_close(tensor, before[name].flip(0), atol=0., rtol=0.)
                    else:
                        torch.testing.assert_close(tensor.flatten(1).norm(dim=1),
                                                   before[name].flatten(1).norm(dim=1),
                                                   atol=1e-5, rtol=1e-5)

    def test_owner_clock_byte_and_context_guards(self):
        model = models.make_model("transformer", 9701)
        tokens = torch.ones(1, 3, dtype=torch.long)
        _, state = model(tokens, owners=("a",))
        with self.assertRaises(ValueError):
            model(tokens, state=state, owners=("other",))
        with self.assertRaises(ValueError):
            model(tokens.float())
        with self.assertRaises(ValueError):
            model(torch.full_like(tokens, 256))
        huge = torch.ones(1, models.MAX_TRANSFORMER_CONTEXT + 1, dtype=torch.long)
        with self.assertRaises(ValueError):
            model(huge)
        with self.assertRaises(ValueError):
            models.intervene(state, "donor", [1])
        with self.assertRaises(ValueError):
            models.intervene(state, "donor", [.5])


if __name__ == "__main__":
    unittest.main()
