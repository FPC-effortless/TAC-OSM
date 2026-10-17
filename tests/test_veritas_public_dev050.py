import json

import pytest

from tac_osm.development.veritas_public_challenge import make_episode, run_episode, run_policy


@pytest.fixture(autouse=True)
def require_optional_veritas():
    pytest.importorskip("investigation_world.operational.runtime")


def faulty_episode():
    return next(e for i in range(100) if (e := make_episode(1, i)).oracle.initial_state["e0.fault"]
                and e.oracle.initial_state["e1.fault"])


def test_public_policy_boundary_contains_no_oracle():
    episode = faulty_episode()
    public = episode.public_payload()
    assert "oracle" not in public
    assert "initial_state" not in json.dumps(public)
    calls = []
    run_policy("service_all_once", public, lambda action, **p: calls.append((action, p)))
    assert [a for a, _ in calls] == ["isolate", "service", "isolate", "service"]


def test_blocked_service_gets_no_repair_credit():
    result = run_episode(faulty_episode(), "service_without_isolation")
    assert result["failures"] == 2
    assert all(e["blocked"] and not e["effect_applied"] for e in result["trace"])
    assert result["verification"]["state"] < 1
    assert result["verification"]["process"] == 0


def test_shared_slot_prevents_simultaneous_isolation():
    from investigation_world.operational.runtime import OperationalRuntime

    runtime = OperationalRuntime(faulty_episode())
    assert runtime.act("isolate", entity=0)["accepted"]
    assert not runtime.act("isolate", entity=1)["accepted"]
    assert not runtime.act("service", entity=1, attempt=1)["accepted"]


def test_repairing_final_state_does_not_erase_trajectory_violation():
    from investigation_world.operational.models import EpisodeSubmission
    from investigation_world.operational.runtime import OperationalRuntime

    runtime = OperationalRuntime(make_episode(1, 0))
    runtime.act("bypass")
    runtime.act("erase")
    result = runtime.submit(EpisodeSubmission())
    assert "no_collateral" in result.invariant_violations
    assert "bypass" in result.forbidden_actions_taken


def test_attempt_numbers_cannot_skip_or_replay_transitions():
    from investigation_world.operational.runtime import OperationalRuntime

    runtime = OperationalRuntime(make_episode(1, 3))
    runtime.act("isolate", entity=0)
    assert not runtime.act("service", entity=0, attempt=2)["accepted"]
    assert runtime.act("service", entity=0, attempt=1)["accepted"]
    runtime.act("isolate", entity=0)
    assert not runtime.act("service", entity=0, attempt=1)["accepted"]
    assert runtime.act("service", entity=0, attempt=2)["accepted"]


def test_verify_retry_resolves_first_attempt_failure_and_replays():
    # Search only this test's development seeds, never a held-out evaluation.
    episode = next(make_episode(1, i) for i in range(200)
                   if run_episode(make_episode(1, i), "inspect_once")["failures"] > 0)
    result = run_episode(episode, "inspect_verify_retry")
    assert result["failures"] == 0
    assert result == run_episode(episode, "inspect_verify_retry")
    assert result["sensing_calls"] > 2
    assert result["total_cost"] > run_episode(episode, "inspect_once")["verification"]["cost_spent"]/10
    assert not result["verification"]["invariant_violations"]


def test_public_identifiers_do_not_control_reference_actions():
    from copy import deepcopy
    from tac_osm.development.veritas_public_challenge import POLICIES

    public = make_episode(1, 0).public_payload()
    changed = deepcopy(public)
    changed["episode_id"] = "renamed"
    changed["world_id"] = "renamed"
    changed["task"]["task_id"] = "renamed"
    changed["task"]["world_id"] = "renamed"
    for policy in POLICIES:
        traces = []
        for payload in (public, changed):
            calls = []
            def act(action, **parameters):
                calls.append((action, parameters))
                return {"fault": 1, "accepted": True}
            run_policy(policy, payload, act)
            traces.append(calls)
        assert traces[0] == traces[1]
