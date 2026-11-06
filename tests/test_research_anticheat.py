"""Adversarial, no-fit unit fixtures for TAC-OSM research anti-cheat.

Constructed records are mechanism tests ONLY, not real-world capability data.
"""
from __future__ import annotations

from dataclasses import dataclass
import itertools
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

import pytest
import torch

from tac_osm.research_anticheat import (
    AuditGateError, AuditLedger, Finding, audit_forward_time,
    audit_input_boundary, audit_partitions, audit_train_only_fit,
    fingerprint, probe_future_invariance, probe_prediction_invariance,
    torch_gradient_diagnostics, trace_torch_modules,
)
from tac_osm.real_robot_transition_contract import RealLoggedWindow

ROBOT_ALLOWED = (
    "past_and_current_measured_states", "strictly_prior_executed_actions",
    "current_measured_state", "current_logged_action",
    "currently_human_controlled",
)


def robot_fixture(next_value=4.0) -> RealLoggedWindow:
    """Tiny contract fixture, not a robot observation or training dataset."""
    zero = (0.0,) * 14
    return RealLoggedWindow(
        observed_history_state=(zero, zero),
        previously_executed_actions=(zero,),
        current_measured_state=zero,
        current_logged_action=zero,
        currently_human_controlled=False,
        next_measured_state=(next_value,) * 14,
        new_intervention_onset=False,
        current_frame_index=1, episode_index=11,
    )


def test_real_robot_inputs_pass_named_boundary():
    s = robot_fixture()
    assert audit_input_boundary(s.model_inputs(), ROBOT_ALLOWED).status == "PASS"


def test_real_robot_full_window_is_not_a_permitted_model_input():
    s = robot_fixture()
    finding = audit_input_boundary({"window": s}, ["window"])
    assert finding.status == "FAIL"
    assert any("next_measured_state" in p for p in finding.evidence["paths"])


def test_nested_future_label_is_rejected():
    finding = audit_input_boundary(
        {"observation": [{"sensor": 1, "nested": {"future_reward": 99}}]},
        ["observation"])
    assert finding.status == "FAIL"
    assert "root.observation[0].nested.future_reward: forbidden field" in finding.evidence["paths"]


def test_unlisted_input_key_is_rejected_even_if_neutral_name():
    finding = audit_input_boundary({"context": 1, "debug_hint": 7}, ["context"])
    assert finding.status == "FAIL"


def test_unknown_object_container_is_fail_closed():
    class Hidden:
        answer = 42
    assert audit_input_boundary({"context": Hidden()}, ["context"]).status == "FAIL"


def test_cyclic_mapping_does_not_hang():
    x = {}
    x["self"] = x
    assert audit_input_boundary({"context": x}, ["context"]).status == "PASS"


def test_future_poison_real_robot_contract_passes():
    a = robot_fixture(4)
    b = robot_fixture(9)
    f = probe_future_invariance(a, b, lambda s: s.model_inputs(),
                                lambda s: s.factual_targets())
    assert f.status == "PASS"
    assert f.evidence["targets_changed"] is True


def test_future_poison_catches_injected_target_path():
    a = robot_fixture(4)
    b = robot_fixture(9)
    f = probe_future_invariance(a, b,
                                lambda s: {"context": s.current_measured_state,
                                           "leaked": s.next_measured_state},
                                lambda s: s.factual_targets())
    assert f.status == "FAIL"


def test_future_poison_without_modified_outcome_is_inconclusive():
    s = robot_fixture(4)
    assert probe_future_invariance(s, s, lambda x: x.model_inputs(),
                                   lambda x: x.factual_targets()).status == "INCONCLUSIVE"


def test_prediction_probe_accepts_identical_permitted_state():
    f = probe_prediction_invariance(lambda x: x["context"] ** 2,
                                    {"context": 2}, {"context": 2})
    assert f.status == "PASS"


def test_prediction_probe_catches_side_channel():
    f = probe_prediction_invariance(lambda x: x["metadata"]["answer"],
                                    {"metadata": {"answer": 0}},
                                    {"metadata": {"answer": 1}})
    assert f.status == "FAIL"


def test_prediction_probe_does_not_misclassify_nondeterminism_as_leakage():
    counter = itertools.count()
    f = probe_prediction_invariance(lambda _: next(counter), 1, 1)
    assert f.status == "INCONCLUSIVE"


def test_split_exact_id_disjointness_passes():
    assert audit_partitions({"FIT": [1, 2], "DEV": [3], "TEST": [4]}).status == "PASS"


def test_duplicate_within_split_fails():
    f = audit_partitions({"FIT": [1, 1], "DEV": [2], "TEST": [3]})
    assert f.status == "FAIL" and f.evidence["duplicate_within"] == ["FIT"]


def test_source_id_overlap_across_splits_fails():
    f = audit_partitions({"FIT": [1, 2], "DEV": [2], "TEST": [3]})
    assert f.status == "FAIL"
    assert f.evidence["overlap_pairs"] == ["FIT/DEV"]


def test_normalizer_fit_only_on_train_passes():
    assert audit_train_only_fit([1, 2], [1, 2, 3]).status == "PASS"


def test_normalizer_fit_on_dev_row_fails():
    assert audit_train_only_fit([1, 2, 3], [1, 2]).status == "FAIL"


def test_forward_chronology_with_embargo_passes():
    f = audit_forward_time({"FIT": [("robot-a", 1.0)],
                            "DEV": [("robot-a", 3.0)],
                            "TEST": [("robot-a", 5.0)]}, embargo=1.0)
    assert f.status == "PASS"


def test_forward_chronology_overlap_is_detected():
    f = audit_forward_time({"FIT": [("robot-a", 3.0)],
                            "DEV": [("robot-a", 2.0)],
                            "TEST": [("robot-a", 5.0)]})
    assert f.status == "FAIL"
    assert f.evidence["violating_entity_boundaries"] > 0


def test_missing_diagnostic_is_not_a_green_gate():
    ledger = AuditLedger("run-1", ["boundary", "poison"])
    ledger.add(Finding("boundary", "PASS", "clean", {}))
    assert ledger.report()["status"] == "INCONCLUSIVE"
    with pytest.raises(AuditGateError):
        ledger.assert_clean()


def test_all_required_diagnostics_can_pass():
    ledger = AuditLedger("run-2", ["boundary", "poison"])
    ledger.add(Finding("boundary", "PASS", "clean", {}))
    ledger.add(Finding("poison", "PASS", "clean", {}))
    assert ledger.report()["status"] == "PASS"
    ledger.assert_clean()


def test_failed_diagnostic_blocks_run():
    ledger = AuditLedger("run-3", ["boundary"])
    ledger.add(Finding("boundary", "FAIL", "leak", {"paths": ["x"]}))
    assert ledger.report()["status"] == "FAIL"
    with pytest.raises(AuditGateError):
        ledger.assert_clean()


def test_missing_gradient_route_is_localized():
    m = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Linear(2, 1))
    out = m[0](torch.ones(1, 2))
    out.sum().backward()
    f = torch_gradient_diagnostics(m, expected_prefixes=["0", "1"])
    assert f.status == "FAIL"
    assert f.evidence["blocks"]["1"]["disconnected"]


def test_active_gradient_route_is_observable():
    m = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Linear(2, 1))
    m(torch.ones(1, 2)).sum().backward()
    f = torch_gradient_diagnostics(m, expected_prefixes=["0", "1"])
    assert f.status == "PASS"
    assert f.evidence["blocks"]["0"]["gradient_l2"] > 0


def test_forward_trace_identifies_nan_origin_and_removes_hooks():
    m = torch.nn.Sequential(torch.nn.Linear(1, 1), torch.nn.ReLU())
    with trace_torch_modules(m) as trace:
        m(torch.tensor([[float("nan")]]))
    assert trace
    assert trace[0]["module"] == "0"
    assert trace[0]["finite"] is False
    assert all(not mod._forward_hooks for mod in m.modules())


def test_digests_are_repeatable_but_not_source_authentication():
    assert fingerprint({"a": [1, 2]}) == fingerprint({"a": [1, 2]})
    assert fingerprint({"a": [1, 2]}) != fingerprint({"a": [1, 3]})


def test_graph_probe_ignores_unused_forbidden_metadata():
    from tac_osm.research_anticheat import trace_torch_noninterference
    m = torch.nn.Linear(1, 1)
    def forward(network, row):
        return network(row["legit"])
    a = {"legit": torch.tensor([[2.0]]), "forbidden_label": 0}
    b = {"legit": torch.tensor([[2.0]]), "forbidden_label": 1}
    f = trace_torch_noninterference(m, forward, a, b)
    assert f.status == "PASS", f


def test_graph_probe_names_first_illicit_consumer_module():
    from tac_osm.research_anticheat import trace_torch_noninterference
    class Illicit(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.good = torch.nn.Linear(1, 1, bias=False)
            self.oracle_reader = torch.nn.Linear(1, 1, bias=False)
            with torch.no_grad():
                self.oracle_reader.weight.fill_(1.0)
        def forward(self, x):
            return self.good(x["sensor"]) + self.oracle_reader(x["hidden"])
    m = Illicit()
    a = {"sensor": torch.tensor([[1.0]]), "hidden": torch.tensor([[0.0]])}
    b = {"sensor": torch.tensor([[1.0]]), "hidden": torch.tensor([[1.0]])}
    f = trace_torch_noninterference(m, lambda network, x: network(x), a, b)
    assert f.status == "FAIL", f
    assert f.evidence["first_changed_module"] == "oracle_reader"


def test_graph_probe_reports_pure_functional_or_external_dependency():
    from tac_osm.research_anticheat import trace_torch_noninterference
    class Functional(torch.nn.Module):
        def forward(self, x):
            return x["hidden"] * 2
    a = {"hidden": torch.tensor([1.0])}
    b = {"hidden": torch.tensor([2.0])}
    f = trace_torch_noninterference(Functional(), lambda network, x: network(x), a, b)
    assert f.status == "FAIL"
    assert f.evidence["first_changed_module"] == "<untraced_functional_or_external_path>"


def test_graph_probe_does_not_greenlight_external_randomness():
    from tac_osm.research_anticheat import trace_torch_noninterference
    import random
    class Randomized(torch.nn.Module):
        def forward(self, _):
            return torch.tensor([random.random()])
    f = trace_torch_noninterference(Randomized(), lambda network, x: network(x), {}, {})
    assert f.status == "INCONCLUSIVE"
