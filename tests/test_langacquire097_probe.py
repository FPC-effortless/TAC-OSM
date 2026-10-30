"""Constructed no-fit source adapter checks; never capability evidence."""
import importlib.util
from pathlib import Path
import sys
from dataclasses import replace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
HERE = ROOT / "research/mathematical-engineering/langacquire097"


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


models = load("langacquire097_models_probe_tests", HERE / "models.py")
probes = load("langacquire097_probe_tests", HERE / "probe_adapter.py")

from tac_osm.research_probe import (  # noqa: E402
    probe_dependencies, probe_entity_isolation, probe_memory,
)


@pytest.mark.parametrize("arm", probes.ARMS)
def test_actual_arm_adapter_state_purity_isolation_and_local_gradients(arm):
    torch.set_num_threads(1)
    model = models.make_model(arm, seed=97001).eval()
    sample = {"tokens": torch.tensor([[65, 66, 67, 68], [70, 71, 72, 73]]),
              "owners": ("constructed-a", "constructed-b")}
    adapter = probes.adapter097(arm)
    memory = probe_memory(model, sample, adapter)
    assert memory["events"] == 3
    assert memory["query_does_not_write_state"]
    assert memory["candidate_permutation"]["status"] == "INCONCLUSIVE_NOT_DECLARED"
    assert memory["state_components"]
    isolation = probe_entity_isolation(model, sample, adapter, tolerance=3e-5)
    assert isolation["status"] == "PASS"
    dependency = probe_dependencies(model, sample, adapter)
    assert dependency["status"] == "MEASURED"
    assert any(row["connected"] and row["finite"]
               for row in dependency["heads"]["nextbyte_logits"].values())
    if arm == "transformer":
        changed = [component for event in memory["timeline"]
                   for component in event["components"].values()
                   if component.get("change_status") == "INCONCLUSIVE_SHAPE_CHANGED"]
        assert changed, "the source KV cache grows; do not hide or pad its shape"
        assert all(component["change_l2"] is None for component in changed)


def test_adapter_rejects_targets_and_an_unreviewed_arm_swap():
    model = models.make_model("tac", seed=97002).eval()
    sample = {"tokens": torch.tensor([[65, 66]]), "owners": ("fixture",)}
    with pytest.raises(ValueError, match="only observed"):
        probes.adapter097("tac").validate(model, {**sample, "target": 67})
    with pytest.raises(ValueError, match="source-reviewed"):
        probes.adapter097("gru").validate(model, sample)
    with pytest.raises(ValueError, match="registered"):
        probes.adapter097("unregistered")


@pytest.mark.parametrize("arm", probes.ARMS)
def test_actual_query_state_derivative_agrees_with_finite_difference(arm):
    """Check a genuine native-state derivative, not an affine stand-in."""
    torch.set_num_threads(1)
    model = models.make_model(arm, seed=97003).double().eval()
    owners = ("constructed-gradient-owner",)
    _, acquired = model(torch.tensor([[65, 66, 67, 68]]), owners=owners)
    leaves = {name: value.detach().clone().requires_grad_(True)
              for name, value in acquired.tensors.items()}
    source = replace(acquired, tensors=leaves)
    query = torch.tensor([[70]])
    logits, _ = model(query, state=source, owners=owners)
    scalar = logits[0, 0, 71]
    gradients = torch.autograd.grad(scalar, tuple(leaves.values()), allow_unused=True)
    candidates = [(float(gradient.abs().max()), name, gradient)
                  for name, gradient in zip(leaves, gradients)
                  if gradient is not None and gradient.numel()]
    maximum, name, gradient = max(candidates, key=lambda row: row[0])
    assert maximum > 1e-8
    index = int(gradient.abs().flatten().argmax())
    epsilon = 1e-5
    values = []
    for sign in (1., -1.):
        changed = {key: value.detach().clone() for key, value in leaves.items()}
        changed[name].reshape(-1)[index] += sign * epsilon
        output, _ = model(query, state=replace(acquired, tensors=changed), owners=owners)
        values.append(float(output[0, 0, 71].detach()))
    numerical = (values[0] - values[1]) / (2 * epsilon)
    analytic = float(gradient.flatten()[index])
    assert abs(numerical - analytic) <= 1e-7 + 1e-4 * abs(analytic)
