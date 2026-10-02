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
    updated, probs = state.write(memory, z, target_entity=entities)
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


def test_address_state_matches_base_verifier_write_signature():
    import inspect
    from tac_osm.integrated_e2e_address import AddressDiagnosisState
    params = list(inspect.signature(AddressDiagnosisState.write).parameters)
    assert params[:4] == ["self", "memory", "z", "target_entity"]


def test_e2e002_contract_is_loadable_and_consistent():
    from tac_osm.contract import load_contract
    contract = load_contract("TACOSM-PLM-INTEGRATED-E2E-002")
    assert contract.check_consistency() == []


def test_e2e002_evaluator_controls_all_have_defined_inputs():
    from scripts.run_integrated_e2e_001 import sample_episode
    from scripts.run_integrated_e2e_002 import evaluate
    model = AddressDiagnosisModel(
        AddressDiagnosisConfig(hidden_dim=40, write_mode="explicit", read_mode="explicit")
    )
    ep = sample_episode(__import__("random").Random(4242))
    for control in ("normal", "no_memory", "shuffle_image", "text_only", "image_only", "audio_only"):
        out = evaluate(model, [ep], control)
        assert set(out) >= {"q2_accuracy", "target_memory_attention", "write_target_address_accuracy"}
        assert all(float(v) == float(v) for v in out.values())
