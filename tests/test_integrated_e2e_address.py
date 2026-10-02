import pytest

torch = pytest.importorskip("torch")
import torch as th

from tac_osm.integrated_e2e_address import (
    AddressDiagnosisConfig,
    AddressDiagnosisModel,
    AddressDiagnosisState,
)


def test_explicit_state_paths_are_one_hot_and_entity_local():
    state = AddressDiagnosisState(8, 4, "explicit", "explicit")
    memory = th.zeros(2, 4, 8)
    z = th.ones(2, 8)
    entities = th.tensor([1, 3])
    updated, probs = state.write(memory, z, entity=entities)
    assert th.allclose(probs[0], th.tensor([0., 1., 0., 0.]))
    assert th.allclose(probs[1], th.tensor([0., 0., 0., 1.]))
    read, attention = state.read(updated, th.zeros(2, 8), entity=entities)
    assert th.allclose(attention, probs)
    assert th.count_nonzero(read[0]) > 0
    assert th.count_nonzero(read[1]) > 0


def test_address_diagnosis_backward_reaches_integrated_surfaces():
    model = AddressDiagnosisModel(
        AddressDiagnosisConfig(hidden_dim=40, write_mode="explicit", read_mode="explicit")
    )
    batch = 4
    steps = 3
    text = th.randint(0, 40, (steps, batch, 7))
    image = th.rand(steps, batch, 1, 16, 16)
    audio = th.randn(steps, batch, 1, 96)
    entities = th.randint(0, 16, (steps, batch))
    payloads = th.randint(0, 2, (steps, batch, 12)).float()
    q = (
        th.randint(0, 16, (batch,)),
        th.randint(0, 12, (batch,)),
        th.randint(0, 12, (batch,)),
        th.randint(0, 4, (batch,)),
        th.randint(0, 2, (batch,)),
    )
    out = model.forward_episode(
        {"text": text, "image": image, "audio": audio},
        entities, payloads, q, q
    )
    out["loss"].backward()
    for name in (
        "text.emb.weight",
        "image.net.0.weight",
        "audio.net.0.weight",
        "state.write_value.weight",
        "casm.selector.0.weight",
        "verifier.0.weight",
        "write_gate.0.weight",
    ):
        param = dict(model.named_parameters())[name]
        assert param.grad is not None
        assert th.isfinite(param.grad).all()
