import pytest

torch = pytest.importorskip("torch")

from tac_osm.integrated_e2e import IntegratedConfig, IntegratedE2EModel


def test_end_to_end_backward_reaches_all_chain_surfaces():
    import torch as th

    model = IntegratedE2EModel(IntegratedConfig(hidden_dim=40))
    batch = 4
    steps = 3
    text = th.randint(0, 40, (steps, batch, 10))
    image = th.rand(steps, batch, 1, 16, 16)
    audio = th.randn(steps, batch, 1, 96)
    entities = th.randint(0, 16, (steps, batch))
    payloads = th.randint(0, 2, (steps, batch, 8)).float()
    q = (
        th.randint(0, 16, (batch,)),
        th.randint(0, 8, (batch,)),
        th.randint(0, 8, (batch,)),
        th.randint(0, 4, (batch,)),
        th.randint(0, 2, (batch,)),
    )

    out = model.forward_episode(
        {"text": text, "image": image, "audio": audio},
        entities,
        payloads,
        q,
        q,
    )
    out["loss"].backward()

    for name in (
        "text.emb.weight",
        "image.net.0.weight",
        "audio.net.0.weight",
        "state.write_entity.weight",
        "state.read_query.weight",
        "casm.selector.0.weight",
        "verifier.0.weight",
        "write_gate.0.weight",
    ):
        param = dict(model.named_parameters())[name]
        assert param.grad is not None
        assert th.isfinite(param.grad).all()


def test_modality_encoders_preserve_expected_shapes():
    import torch as th

    model = IntegratedE2EModel()
    z, _, _, _ = model.encode(
        th.randint(0, 40, (4, 10)),
        th.rand(4, 1, 16, 16),
        th.randn(4, 1, 96),
    )
    assert z.shape == (4, 40)


def test_preregistered_compositions_are_strictly_disjoint():
    from scripts.run_integrated_e2e_001 import HELDOUT, TRAIN_COMBOS

    assert set(HELDOUT).isdisjoint(TRAIN_COMBOS)


def test_pre_action_query_has_no_outcome_argument():
    import inspect

    model = IntegratedE2EModel()
    names = list(inspect.signature(model.query).parameters)
    assert names == ["memory", "entity", "i", "j", "op"]


def test_latent_state_is_complementary_12_bit():
    from tac_osm.integrated_e2e import LATENT_BITS
    assert LATENT_BITS == 12
