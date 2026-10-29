"""Constructed regression checks; these establish no language capability."""
from dataclasses import replace

import pytest
import torch

from scripts.langdecode057_validation import BatchedValidationTAC057
from incremental import IncrementalTAC043, SHAPES
from factorial_depth import FactorialDepthControl040


def engines(dtype=torch.float32, mode="cap_all"):
    torch.manual_seed(57057)
    torch.set_num_threads(2)
    model = FactorialDepthControl040(state_width=48, upper_graph=False).to(dtype).eval()
    return IncrementalTAC043(model, mode), BatchedValidationTAC057(model, mode)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("mode", ["native", "cap_all"])
def test_exact_transition_and_snapshot_interoperability(tmp_path, dtype, mode):
    reference, batched = engines(dtype, mode)
    owners = ("document-a", "document-b")
    tokens = torch.randint(0, 256, (2, 145))
    masks = torch.arange(145)[None, :] % 3 != torch.arange(2)[:, None]
    y, state = reference.prefill(tokens[:, :129], owners=owners, write_mask=masks[:, :129])
    z, other = batched.prefill(tokens[:, :129], owners=owners, write_mask=masks[:, :129])
    assert torch.equal(y, z)
    for t in range(129, 145):
        before = {name: value.clone() for name, value in other.tensors.items()}
        y, state = reference.step(tokens[:, t], state, owners=owners, write_mask=masks[:, t])
        z, next_state = batched.step(tokens[:, t], other, owners=owners, write_mask=masks[:, t])
        assert torch.equal(y, z)
        assert all(torch.equal(before[name], value) for name, value in other.tensors.items())
        assert all(torch.equal(state.tensors[name], next_state.tensors[name]) for name in SHAPES)
        assert next_state.tensor_bytes() == 2 * 976 * torch.tensor([], dtype=dtype).element_size()
        other = next_state
    path = tmp_path / "original.pt"
    reference.save_state(state, path)
    restored = batched.load_state(path, owners=owners)
    path2 = tmp_path / "batched.pt"
    batched.save_state(restored, path2)
    restored = reference.load_state(path2, owners=owners)
    y, _ = reference.step(tokens[:, 0], state, owners=owners)
    z, _ = batched.step(tokens[:, 0], restored, owners=owners)
    assert torch.equal(y, z)


@pytest.mark.parametrize("field", SHAPES)
@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), -float("inf")])
def test_each_nonfinite_field_is_rejected(field, bad_value):
    reference, batched = engines()
    state = reference.start(("a",))
    state.tensors[field].flatten()[0] = bad_value
    for engine in (reference, batched):
        with pytest.raises(ValueError, match="state tensor invalid: " + field):
            engine.validate(state, ("a",))


def test_boundary_guards_remain_and_invalid_outputs_are_rejected():
    reference, batched = engines()
    state = reference.start(("a",))
    for engine in (reference, batched):
        for bad in (replace(state, clock=-1), replace(state, clock=2**63-1),
                    replace(state, tensors={}), replace(state, position_mode="other")):
            with pytest.raises(ValueError):
                engine.step(torch.tensor([1]), bad, owners=("a",))
        with pytest.raises(ValueError, match="owner"):
            engine.validate(state, ("b",))
        for token in (torch.tensor([-1]), torch.tensor([256]), torch.tensor([1.]), torch.tensor([[1]])):
            with pytest.raises(ValueError):
                engine.step(token, state, owners=("a",))
        for bad_tensor in (torch.zeros(1), state.tensors["wave"].double(),
                           state.tensors["wave"].clone().requires_grad_()):
            bad = replace(state, tensors={**state.tensors, "wave": bad_tensor})
            with pytest.raises(ValueError):
                engine.validate(bad, ("a",))
    hook = batched.model.embedding.register_forward_hook(lambda *args: None)
    with pytest.raises(ValueError, match="hook"):
        batched.validate(state, ("a",))
    hook.remove()
    batched.model.train()
    with pytest.raises(ValueError, match="eval"):
        batched.validate(state, ("a",))
    batched.model.eval()
    with torch.no_grad():
        batched.model.ffn_out.bias.add_(1)
    with pytest.raises(ValueError, match="weights"):
        batched.validate(state, ("a",))


def test_nonfinite_transition_output_still_rejected():
    _, batched = engines()
    state = batched.start(("a",))
    advance = batched._advance

    def poisoned(*args):
        logits, result = advance(*args)
        result.tensors["wave"].fill_(float("nan"))
        return logits, result

    batched._advance = poisoned
    with pytest.raises(ValueError, match="wave"):
        batched.step(torch.tensor([1]), state, owners=("a",))
