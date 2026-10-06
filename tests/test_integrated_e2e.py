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
    from tac_osm.integrated_e2e_benchmark import HELDOUT, TRAIN_COMBOS

    assert set(HELDOUT).isdisjoint(TRAIN_COMBOS)


def test_pre_action_query_has_no_outcome_argument():
    import inspect

    model = IntegratedE2EModel()
    names = list(inspect.signature(model.query).parameters)
    assert names == ["memory", "entity", "i", "j", "op"]


def test_latent_state_is_complementary_12_bit():
    from tac_osm.integrated_e2e import LATENT_BITS
    assert LATENT_BITS == 12


def test_all_registered_query_pairs_are_canonical():
    from tac_osm.integrated_e2e_benchmark import TRAIN_COMBOS, HELDOUT
    assert all(i < j for _, i, j in TRAIN_COMBOS)
    assert all(i < j for _, i, j in HELDOUT)
    assert all((op, j, i) not in TRAIN_COMBOS for op, i, j in HELDOUT)


def test_heldout_queries_cross_modal():
    from tac_osm.integrated_e2e_benchmark import HELDOUT
    assert all(
        (i < 4) != (j < 4)
        or (4 <= i < 8) != (4 <= j < 8)
        or (8 <= i < 12) != (8 <= j < 12)
        for _, i, j in HELDOUT
    )


def test_verified_feedback_write_is_entity_addressed():
    import torch as th

    from tac_osm.integrated_e2e import PersistentKeyValueState

    state = PersistentKeyValueState(hidden=8, entity_count=4)
    memory = th.zeros(1, 4, 8)
    z = th.ones(1, 8)
    updated, probs = state.write(
        memory, z, strength=th.ones(1), target_entity=th.tensor([2])
    )
    assert probs.shape == (1, 4)
    assert probs[0].tolist() == [0.0, 0.0, 1.0, 0.0]
    assert th.count_nonzero(updated[0, 0]) == 0
    assert th.count_nonzero(updated[0, 1]) == 0
    assert th.count_nonzero(updated[0, 2]) == 8
    assert th.count_nonzero(updated[0, 3]) == 0


def test_verifier_consumes_only_pre_verification_features():
    import torch as th

    model = IntegratedE2EModel()
    assert model.verifier[0].in_features == 41
    memory = model.state.initial(1, th.device("cpu"))
    q = {
        "entity": th.tensor([0]),
        "read": th.zeros(1, 40),
        "action": th.zeros(1),
    }
    # The call signature requires an observed environment outcome separately;
    # it must not be embedded in verifier_input itself.
    updated, verifier = model.post_action_update(memory, q, th.tensor([1]))
    assert verifier.shape == (1, 2)
    assert th.isfinite(updated).all()
