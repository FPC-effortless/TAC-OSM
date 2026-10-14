"""DEV-017 CPU learned-state, causal-control and sensing boundary checks."""
from pathlib import Path
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from run_learned_schema_dev import (  # noqa: E402
    matching_models, public_stream_tensors, public_queries, labels,
    select_entity, services_from_probabilities, closed_loop,
)
from tac_osm.development.learned_schema_state import MaintenancePredictor  # noqa: E402
from tac_osm.development.schema_maintenance import make_pair  # noqa: E402


def inputs():
    cases = make_pair(43, 32, 16, "pending_cancel", "at")
    stream = public_stream_tensors([c.observations for c in cases])
    query = public_queries([c.tick for c in cases], [c.horizon for c in cases])
    return cases, stream, query


def test_41_scalar_slots_and_global_are_learned_grus():
    m = MaintenancePredictor("schema41")
    cases, stream, query = inputs()
    state = m.encode(*stream)
    assert state.shape == (2, 41)
    logits = m.decode(state, query)
    F.cross_entropy(logits.reshape(-1, 2), labels(cases).reshape(-1)).backward()
    for p in (m.slot_gru.weight_hh, m.global_gru.weight_ih):
        assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0


def test_causal_groups_have_identical_initial_tensors():
    _, hashes = matching_models(43)
    assert hashes["schema41"] == hashes["schema41_encoder"]
    assert len({hashes[a] for a in hashes if a.startswith("tac")}) == 1


def test_tac_memory_only_has_writer_gradient_with_frozen_encoder():
    m = MaintenancePredictor("tac_memory")
    cases, stream, query = inputs()
    logits = m(*stream, query)
    F.cross_entropy(logits.reshape(-1, 2), labels(cases).reshape(-1)).backward()
    assert m.event_encoder[0].weight.grad is None
    grad = m.reader.memory.write.weight.grad
    assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0
    assert m.reader.compute.selector[-1].weight.grad is not None


def test_tac_encoder_only_has_encoder_gradient_with_frozen_writer():
    m = MaintenancePredictor("tac_encoder")
    cases, stream, query = inputs()
    logits = m(*stream, query)
    F.cross_entropy(logits.reshape(-1, 2), labels(cases).reshape(-1)).backward()
    assert m.event_encoder[0].weight.grad is not None
    assert m.reader.memory.write.weight.grad is None


def test_schema_encoder_control_freezes_recurrent_updates():
    m = MaintenancePredictor("schema41_encoder")
    cases, stream, query = inputs()
    logits = m(*stream, query)
    F.cross_entropy(logits.reshape(-1, 2), labels(cases).reshape(-1)).backward()
    assert m.slot_gru.weight_hh.grad is None
    assert m.global_gru.weight_hh.grad is None
    assert m.event_encoder[0].weight.grad is not None


def test_tac_state_is_exactly_768_scalars():
    m = MaintenancePredictor("tac_joint")
    _, stream, _ = inputs()
    state = m.encode(*stream)
    assert state.shape == (2, 4, 8, 24)
    assert state[0].numel() == 768


def test_zero_state_has_equal_counterfactual_forecasts():
    _, stream, query = inputs()
    for arm in ("schema41", "tac_joint"):
        m = MaintenancePredictor(arm).eval()
        with torch.no_grad():
            state = m.encode(*stream)
            output = m.decode(torch.zeros_like(state), query)
        torch.testing.assert_close(output[0], output[1], atol=1e-6, rtol=1e-6)
        assert torch.equal(output.argmax(-1)[0], output.argmax(-1)[1])


def test_sensing_selection_changes_with_learned_beliefs():
    low = [[.01, 0., .01] for _ in range(8)]
    assert select_entity(low, []) is None
    left, right = [p.copy() for p in low], [p.copy() for p in low]
    left[2][0], right[6][0] = .5, .5
    assert select_entity(left, []) == 2
    assert select_entity(right, []) == 6
    assert select_entity(left, [2]) is None


def test_service_choice_charges_failure_and_service_predictions():
    p = [[.1, 0., .1] for _ in range(8)]
    p[3][0] = .8
    assert services_from_probabilities(p) == (3,)
    p[3][1] = .9
    assert services_from_probabilities(p) == ()


def test_closed_loop_only_ingests_chosen_public_receipts():
    m = MaintenancePredictor("schema41").eval()
    cases, stream, query = inputs()
    with torch.no_grad():
        state = m.encode(*stream)
        selected = closed_loop(m, state, query, cases, "selected")
        random = closed_loop(m, state, query, cases, "matched_random",
                             [x["inspections"] for x in selected])
    for episode, a, b in zip(cases, selected, random):
        assert a["inspections"] <= 2 and a["inspections"] == b["inspections"]
        assert len(set(a["inspected_entities"])) == len(a["inspected_entities"])
        for receipt in a["inspection_receipts"]:
            assert receipt["value"] == episode.current_targets[receipt["entity"]]
            assert receipt["kind"] == "inspect"
