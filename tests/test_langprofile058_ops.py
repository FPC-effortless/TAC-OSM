"""Constructed CPU-only instrumentation checks, NOT language capability evidence."""
from types import SimpleNamespace

import pytest
import torch

from scripts.langprofile058_ops import (
    operator_group, summarize_events, wall_samples, profile_stage,
    exact_repeated_state, STAGES, REPETITIONS, PROFILE_REPETITIONS,
)
from scripts.langdecode057_validation import BatchedValidationTAC057
from incremental import SHAPES
from factorial_depth import FactorialDepthControl040


@pytest.mark.parametrize("operator,group", [
    ("aten::_linalg_solve_ex", "small_matrix_factorization_and_solve"),
    ("aten::addmm", "matrix_projections_and_products"),
    ("aten::cat", "tensor_allocation_and_materialization"),
    ("aten::_softmax", "nonlinearities_and_normalization"),
    ("aten::add", "other_aten_ops"),
    ("profiler::_record_function_exit", "not_aten"),
])
def test_operator_classification_is_nonoverlapping(operator, group):
    assert operator_group(operator) == group


def fake_event(key, cpu, count=1):
    return SimpleNamespace(key=key, self_cpu_time_total=cpu, count=count,
                           self_cpu_memory_usage=0)


def test_self_cpu_accounting_excludes_non_aten():
    events = [fake_event("aten::addmm", 120, 3),
              fake_event("aten::_linalg_solve_ex", 70),
              fake_event("aten::cat", 10),
              fake_event("ProfilerStep*", 400)]
    report = summarize_events(events)
    assert report["aten_self_cpu_us"] == 200
    assert report["all_profiled_self_cpu_us"] == 600
    assert report["aten_calls"] == 5
    assert len(report["top_aten_operators"]) == 3
    assert report["category_self_cpu_share"]["matrix_projections_and_products"]["share_of_aten_self_cpu"] == .6
    assert sum(x["share_of_aten_self_cpu"] for x in report["category_self_cpu_share"].values()) == pytest.approx(1.0)


def test_invalid_profiler_events_rejected_without_claim():
    for events in ([], [fake_event("aten::add", -1)],
                   [fake_event("aten::add", float("nan"))],
                   [fake_event("aten::add", 0)],
                   [fake_event("ProfilerStep*", 10)]):
        with pytest.raises(ValueError):
            summarize_events(events)


def test_wall_clock_trials_have_frozen_unprofiled_counts():
    seen = []
    result = wall_samples(lambda: seen.append(1))
    assert len(seen) == REPETITIONS * 7 + 2
    assert result["warmups"] == 2
    assert result["repeats"] == 112
    assert len(result["raw_ms"]) == 112
    assert result["p95_ms"] >= result["median_ms"] >= 0


def test_profiler_does_not_substitute_for_wall_measurement():
    x = torch.tensor([1.0])
    data = profile_stage(lambda: torch.sin(x))
    assert data["profile_calls"] == PROFILE_REPETITIONS
    assert data["aten_self_cpu_us"] > 0
    assert "aten::sin" in [r["operator"] for r in data["top_aten_operators"]]
    assert "median_ms" not in data
    assert STAGES == ("step_with_guards", "transition_only", "validate_only")


def test_unmodified_advance_bitwise_and_same_state_at_clock():
    torch.manual_seed(58058)
    m = FactorialDepthControl040(state_width=48, upper_graph=False).eval()
    engine = BatchedValidationTAC057(m, "cap_all")
    original_state = engine.start(("constructed-owner",))
    token = torch.tensor([97], dtype=torch.long)
    mask = torch.ones_like(token, dtype=torch.bool)
    with torch.inference_mode():
        a, sa = engine.step(token, original_state, owners=("constructed-owner",))
        b, sb = engine._advance(token, original_state, mask)
        assert torch.equal(a, b)
        exact_repeated_state(sa, sb)
        engine.step(token, original_state, owners=("constructed-owner",))
    assert all(torch.count_nonzero(original_state.tensors[k]) == 0 for k in SHAPES)
    assert sa.tensor_bytes() == 3904


def test_frozen_math_transition_is_inherited_unmodified():
    from incremental import IncrementalTAC043
    assert BatchedValidationTAC057._advance is IncrementalTAC043._advance
    assert BatchedValidationTAC057.step is IncrementalTAC043.step
    assert BatchedValidationTAC057.prefill is IncrementalTAC043.prefill
