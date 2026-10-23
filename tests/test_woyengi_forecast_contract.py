"""Synthetic contract tests only; not evidence of real-world forecast skill."""
from copy import deepcopy

import pytest

from tac_osm.woyengi_forecast_contract import (
    SCHEMA, prediction_response, validate_prediction_request,
)


def sample():
    return {
        "schema_version": SCHEMA, "request_id": "r1", "trace_id": "trace-1",
        "principal_id": "actor:1", "workspace_id": "w:1", "entity_id": "robot:1",
        "feature_schema_id": "physical14:v1", "model_id": "me062:untrained",
        "checkpoint_sha256": "a" * 64,
        "authorization_receipt_id": "woyengi-authority-receipt-1",
        "reconstruction_receipt_id": "woyengi-reconstruction-1",
        "state_revision": "event-head-42",
        "valid_at": "2026-10-22T12:00:01Z",
        "recorded_at": "2026-10-22T12:00:02Z",
        "observed_history": [
            {"valid_at": "2026-10-22T12:00:00Z",
             "recorded_at": "2026-10-22T12:00:00Z",
             "provenance_ref": "event:a", "state": [0.] * 14,
             "executed_action": None, "human_takeover": False},
            {"valid_at": "2026-10-22T12:00:01Z",
             "recorded_at": "2026-10-22T12:00:02Z",
             "provenance_ref": "event:b", "state": [1.] * 14,
             "executed_action": [0.] * 14, "human_takeover": False},
        ],
        "candidate_scenarios": [
            {"scenario_id": "candidate:hold", "proposed_actions": [[0.] * 14]},
            {"scenario_id": "candidate:move", "proposed_actions": [[1.] * 14]},
        ],
    }


def test_valid_transport_and_non_authoritative_reply():
    r = sample()
    digest = validate_prediction_request(r)
    assert len(digest) == 64 and digest == validate_prediction_request(deepcopy(r))
    reply = prediction_response(r, forecast_summaries=[
        {"scenario_id": "candidate:hold", "mean_error": None},
        {"scenario_id": "candidate:move", "mean_error": None},
    ], model_replay_commit="b"*40)
    assert reply["execute"] is False
    assert reply["authority"] == "none"
    assert reply["state_revision"] == "event-head-42"
    assert reply["input_sha256"] == digest


@pytest.mark.parametrize("mutate", [
    lambda r: r["observed_history"][-1].update(recorded_at="2026-10-22T12:00:03Z"),
    lambda r: r["observed_history"][-1].update(valid_at="2026-10-22T12:00:03Z"),
    lambda r: r["observed_history"][-1].update(state=[float("nan")]*14),
    lambda r: r["observed_history"][-1].update(provenance_ref=""),
    lambda r: r["candidate_scenarios"].append(r["candidate_scenarios"][0]),
    lambda r: r.update(authorization_receipt_id=""),
    lambda r: r.update(feature_schema_id=""),
    lambda r: r["candidate_scenarios"][0].update(proposed_actions=[[0.]*13]),
])
def test_reject_invalid_and_future_information(mutate):
    r = sample()
    mutate(r)
    with pytest.raises(ValueError):
        validate_prediction_request(r)


def test_reject_reply_with_wrong_scenario_order():
    r = sample()
    with pytest.raises(ValueError):
        prediction_response(r, forecast_summaries=[
            {"scenario_id": "candidate:move"},
            {"scenario_id": "candidate:hold"},
        ], model_replay_commit="b"*40)
