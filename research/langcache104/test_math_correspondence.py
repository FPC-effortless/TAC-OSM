"""Original-source versus independent full scalar graph; no fit/evaluation.

This test needs a source checkout at the explicit --environment path or sibling
TAC-OSM directory. The exact reviewed commit is recorded separately in SOURCE_MAP.
"""
import dataclasses
from copy import deepcopy
import importlib.util
import os
from pathlib import Path
import sys
import unittest

from research.langcache104.math_reference import forward_reference


ROOT = Path(__file__).resolve().parents[2]
SOURCE = Path(os.environ.get("LANGCACHE104_TAC_SOURCE",
                            os.environ.get("TAC_OSM_SOURCE_DIR", ROOT)))
HAVE_SOURCE = (SOURCE/"research/langcache104/model.py").is_file()
HAVE_TORCH = importlib.util.find_spec("torch") is not None


@unittest.skipUnless(HAVE_SOURCE and HAVE_TORCH, "explicit original source + torch required")
class OriginalSourceParity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(SOURCE))
        import torch
        from research.langcache104.model import Config104, DeepCacheTAC104
        cls.torch, cls.Config, cls.Model = torch, Config104, DeepCacheTAC104

    def assert_nested_close(self, actual, expected, path="", tolerance=2e-12):
        if isinstance(expected, list):
            self.assertEqual(len(actual), len(expected), path)
            for i, (a, e) in enumerate(zip(actual, expected)):
                self.assert_nested_close(a, e, path+"/"+str(i), tolerance)
        else:
            self.assertLessEqual(abs(actual-expected), tolerance, path)

    def make_model(self, rotary=True, rope=True):
        self.torch.manual_seed(104)
        config = self.Config(d_model=8, n_layers=2, n_heads=1, key_dim=2,
                             value_dim=2, write_rank=2, ffn_dim=12, window=3,
                             chunk_size=2, rotary=rotary, local_rope=rope)
        return config, self.Model(config).double()

    @staticmethod
    def export_parameters(model):
        parameters = {k: v.detach().tolist() for k, v in model.state_dict().items()}
        # Exported lists do not automatically preserve the source's tied
        # embedding/output parameter identity. Preserve it before perturbing.
        if model.config.tied_embedding:
            parameters["output.weight"] = parameters["embedding.weight"]
        return parameters

    @staticmethod
    def reference_objective(logits):
        return sum(x*.13*(i+1)*((j % 11)-5)
                   for i, row in enumerate(logits) for j, x in enumerate(row))

    def source_objective(self, logits):
        torch = self.torch
        weights = torch.tensor([[.13*(i+1)*((j % 11)-5) for j in range(256)]
                                for i in range(logits.shape[1])], dtype=logits.dtype)
        return (logits[0]*weights).sum()

    @staticmethod
    def largest_coordinate(tensor):
        shape = tensor.shape
        flat = int(tensor.detach().abs().reshape(-1).argmax())
        coordinate = []
        for size in reversed(shape):
            coordinate.append(flat % size)
            flat //= size
        return tuple(reversed(coordinate))

    @staticmethod
    def perturb(values, coordinate, delta):
        target = values
        for i in coordinate[:-1]:
            target = target[i]
        target[coordinate[-1]] += delta

    def assert_derivative(self, measured, expected, name):
        self.assertLessEqual(abs(measured-expected), 2e-8+2e-4*abs(expected), name)

    def compare(self, model, config, tokens, mask, source_state=None, ref_state=None):
        torch = self.torch
        observed = torch.tensor([tokens], dtype=torch.long)
        writes = torch.tensor([mask], dtype=torch.bool)
        parameters = self.export_parameters(model)
        expected, ref_state = forward_reference(tokens, parameters, dataclasses.asdict(config),
                                               state=ref_state, write_mask=mask)
        actual, source_state = model(observed, state=source_state, owners=("stream-a",),
                                     write_mask=writes, mode="chunk")
        self.assert_nested_close(actual[0].detach().tolist(), expected, "logits")
        self.assertEqual(source_state.clock, ref_state["clock"])
        for i, layer in enumerate(source_state.layers):
            for name, tensor in layer.tensors().items():
                self.assert_nested_close(tensor[0].detach().tolist(), ref_state["layers"][i][name],
                                         "state/"+str(i)+"/"+name)
        return source_state, ref_state

    def test_all_layers_logits_and_carried_state_cold_and_continued(self):
        config, model = self.make_model()
        s, r = self.compare(model, config, [1, 250, 0, 72, 31], [True, False, True, True, False])
        self.compare(model, config, [48, 49, 50, 51], [False, True, False, True], s, r)

    def test_no_rotary_and_no_rope_branch(self):
        config, model = self.make_model(rotary=False, rope=False)
        self.compare(model, config, [4, 5, 6, 7, 8], [True]*5)

    def test_one_reference_finite_difference_coordinate_per_named_parameter(self):
        # This samples a max-absolute-gradient coordinate in each named tensor;
        # it is not a proof for every element, model scale, or input history.
        config, model = self.make_model()
        tokens = [11, 12, 13, 14]
        logits, _ = model(self.torch.tensor([tokens]), owners=("stream-a",))
        self.source_objective(logits).backward()
        parameters = self.export_parameters(model)
        h = 1e-5
        for name, parameter in model.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            coordinate = self.largest_coordinate(parameter.grad)
            expected = float(parameter.grad[coordinate])
            plus, minus = deepcopy(parameters), deepcopy(parameters)
            self.perturb(plus[name], coordinate, h)
            self.perturb(minus[name], coordinate, -h)
            upper = self.reference_objective(forward_reference(tokens, plus, dataclasses.asdict(config))[0])
            lower = self.reference_objective(forward_reference(tokens, minus, dataclasses.asdict(config))[0])
            self.assert_derivative((upper-lower)/(2*h), expected, name)

    def test_one_reference_finite_difference_coordinate_per_incoming_state_tensor(self):
        config, model = self.make_model()
        parameters = self.export_parameters(model)
        _, source_state = model(self.torch.tensor([[21, 22]]), owners=("stream-a",))
        _, reference_state = forward_reference([21, 22], parameters, dataclasses.asdict(config))
        state_layers = tuple(type(layer)(**{name: tensor.detach().clone().requires_grad_(True)
                                           for name, tensor in layer.tensors().items()})
                             for layer in source_state.layers)
        source_state = dataclasses.replace(source_state, layers=state_layers)
        tokens = [23, 24]
        logits, _ = model(self.torch.tensor([tokens]), state=source_state, owners=("stream-a",))
        self.source_objective(logits).backward()
        h = 1e-5
        for index, layer in enumerate(source_state.layers):
            for name, tensor in layer.tensors().items():
                self.assertIsNotNone(tensor.grad, str(index)+"/"+name)
                coordinate = self.largest_coordinate(tensor.grad)
                expected = float(tensor.grad[coordinate])
                # The source state includes a batch axis; the independent
                # single-stream reference deliberately does not.
                plus, minus = deepcopy(reference_state), deepcopy(reference_state)
                self.perturb(plus["layers"][index][name], coordinate[1:], h)
                self.perturb(minus["layers"][index][name], coordinate[1:], -h)
                upper = self.reference_objective(forward_reference(tokens, parameters,
                    dataclasses.asdict(config), state=plus)[0])
                lower = self.reference_objective(forward_reference(tokens, parameters,
                    dataclasses.asdict(config), state=minus)[0])
                self.assert_derivative((upper-lower)/(2*h), expected, str(index)+"/"+name)


if __name__ == "__main__":
    unittest.main()
