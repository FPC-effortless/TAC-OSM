"""DEV-052B meaningful boundary, dynamics and end-to-end integrity gates."""
import copy
import json
from pathlib import Path

import pytest
import torch

from tac_osm.development.learned_belief_world import (
    BASE, SHIFTS, ARMS, NEURAL, PublicState, Controller, NativeWorld,
    BeliefModel, make_world, bellman, posterior, collect, estimate,
    tensorize, fit, rollout, strip_timing,
)


def test_execution_protocol_matches_implemented_dimensions_and_matrix():
    proto = json.loads((Path(__file__).resolve().parents[1]/"research/dev052b/execution-protocol.json").read_text())
    assert tuple(proto["arms"]) == ARMS
    assert tuple(proto["neural_arms"]) == NEURAL
    assert proto["training"]["event_dim"] == len(PublicState(0).token()) == 12
    assert proto["model_seeds"] == [7201, 7202, 7203, 7204, 7205]
    assert len({s["seed"] for s in proto["splits"].values()}) == 5


def test_new_world_namespace_is_disjoint_and_public_payload_has_no_rates():
    ep, params, _ = make_world(5521004, 0)
    assert params == BASE
    public = ep.public_payload()
    assert "oracle" not in public
    assert "initial_state" not in str(public)
    assert "repairs" not in str(public)
    assert "accuracy" not in str(public)
    assert ep.oracle.required_action_counts == {}
    assert ep.episode_id.startswith("TAC-DEV052B-")
    assert make_world(5521003, 0)[0].episode_id != ep.episode_id


def test_shift_families_include_uncertain_third_repair_and_native_absorbing_health():
    seen_fail = False
    seen_repaired = False
    for index in range(40):
        world = NativeWorld(5521005, index, True)
        assert world.rates == SHIFTS[index % 4]
        for entity in (0, 1):
            previous = world.fault(entity)
            for attempt in (1, 2, 3):
                assert world.act("isolate", entity)["accepted"]
                assert world.act("service", entity, attempt)["accepted"]
                current = world.fault(entity)
                assert current <= previous
                seen_repaired |= previous == 1 and current == 0
                previous = current
            seen_fail |= current == 1
    assert seen_fail and seen_repaired


def test_constraints_are_enforced_by_native_runtime_and_charged():
    world = NativeWorld(5521004, 9)
    assert world.act("service", 0, 1)["accepted"] is False
    assert world.act("isolate", 0)["accepted"]
    assert world.act("isolate", 1)["accepted"] is False
    assert world.runtime.budget.spent == 14
    assert sum(e.blocked for e in world.runtime.events) == 2


def test_paid_noisy_sensor_is_repeatable_and_bounded():
    a, b = NativeWorld(5521005, 2, True), NativeWorld(5521005, 2, True)
    left = [a.act("inspect", 0)["fault"] for _ in range(4)]
    right = [b.act("inspect", 0)["fault"] for _ in range(4)]
    assert left == right
    assert a.runtime.budget.spent == 4
    assert set(left) <= {0, 1}


def test_bellman_uses_uncertain_third_repair_and_inspection_value():
    assert posterior(.5, 1, .85) == pytest.approx(.85)
    assert bellman(.5, 0, 0, BASE)[1] == 1
    assert bellman(0, 0, 0, BASE)[1] == 0
    assert bellman(1, 2, 4, (.85, (.75, .8, 0.)))[1] == 0
    assert bellman(1, 2, 4, BASE)[1] == 2
    assert bellman(1, 3, 4, BASE) == (4., 0)


def test_blocked_receipt_does_not_commit_and_other_entity_bank_is_unchanged():
    model = BeliefModel(7201, "joint")
    c = Controller("joint", BASE, model)
    for e in (0, 1):
        c.begin(e, {"reported_fault": 1})
    before = {e: h.clone() for e, h in c.banks.items()}
    assert not c.observe(0, "service", {"accepted": False})
    assert torch.equal(c.banks[0], before[0])
    assert c.states[0].attempts == 0
    c.observe(0, "inspect", {"accepted": True, "fault": 1})
    assert torch.equal(c.banks[1], before[1])
    assert not torch.equal(c.banks[0], before[0])


def test_policy_public_boundary_is_invariant_to_uninspected_hidden_outcomes():
    c1, c2 = (Controller("joint", BASE, BeliefModel(7201, "joint")) for _ in range(2))
    c1.begin(0, {"reported_fault": 1})
    c2.begin(0, {"reported_fault": 1})
    for kind, receipt in [("inspect", {"fault": 1}), ("isolate", {"accepted": True}),
                          ("service", {"accepted": True})]:
        c1.observe(0, kind, receipt)
        c2.observe(0, kind, copy.deepcopy(receipt))
    assert c1.choose(0) == c2.choose(0)
    assert c1.predictions == c2.predictions
    # The model receives neither rates nor initial/future fault bits in events.
    assert len(c1.history[0][-1]) == 12


def test_training_rejects_evaluation_worlds_and_keeps_data_budget():
    with pytest.raises(ValueError, match="evaluation"):
        collect({"kind": "in_regime", "seed": 5521004, "worlds": 1})
    seq, sensors = collect({"kind": "train", "seed": 5521001, "worlds": 2})
    assert len(seq) == 4 and len(sensors) == 16
    assert all(len(s) == 11 for s in seq)
    params, table = estimate(seq, sensors)
    assert .5 < params[0] < 1 and len(params[1]) == 3 and table


def test_full_bounded_rollouts_have_native_cost_and_no_resource_violations():
    spec = {"seed": 5521004, "worlds": 1}
    for arm in ARMS:
        model = BeliefModel(7201, arm) if arm in NEURAL else None
        row = rollout(spec, 0, arm, BASE, model)
        assert row["cost"] == pytest.approx(row["native_spent"]/10 + 4*row["failures"])
        assert row["blocked"] == row["violations"] == 0
        assert len(row["calls"]) <= 20
        assert strip_timing(row) == strip_timing(rollout(spec, 0, arm, BASE, model))


def test_recurrent_state_preserves_order_and_last_event_control_forgets_it():
    a, b = torch.tensor([PublicState(0).token("inspect", {"fault": bit}) for bit in (0, 1)]), None
    b = a.flip(0)
    last = torch.tensor([PublicState(0).token("service", {"accepted": True})])
    for arm in ("joint", "last_event"):
        model = BeliefModel(7201, arm)
        states = []
        for seq in (a, b):
            h = torch.zeros(1, 32)
            for token in seq:
                h = model.write(token.unsqueeze(0), h)
            states.append(model.write(last, h))
        assert torch.equal(states[0], states[1]) == (arm == "last_event")


def test_frozen_writer_training_changes_heads_without_changing_writer():
    torch.set_num_threads(1)
    seq, sensors = collect({"kind": "train", "seed": 5521001, "worlds": 2})
    model = BeliefModel(7201, "frozen_writer")
    before = {k: v.clone() for k, v in model.state_dict().items()}
    fit(model, tensorize(seq), sum(sensors)/len(sensors), 2, 7201)
    assert all(torch.equal(before[k], v) for k, v in model.state_dict().items() if k.startswith("writer."))
    assert not torch.equal(before["fault_head.weight"], model.fault_head.weight)
