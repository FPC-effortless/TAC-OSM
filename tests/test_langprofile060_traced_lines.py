"""Constructed sys.settrace source-line attribution safeguards; not speed evidence."""
import sys
import pytest
from scripts.langprofile060_traced_lines import trace_lines, SAMPLES, WARMUPS
from incremental import IncrementalTAC043


def _arithmetic_probe(x):
    y=x+1
    z=y*y
    return z


def test_source_trace_reports_real_python_line_events_and_cleans_up():
    before=sys.gettrace()
    result=trace_lines(_arithmetic_probe,lambda:_arithmetic_probe(3),count=4)
    assert sys.gettrace() is before
    assert result["status"]=="TRACED_PYTHON_LINES_NOT_PRODUCTION_LATENCY"
    assert result["calls"]==4 and result["line_events"]>=8
    assert all(r["line"]>0 and r["samples"]>=4
               for r in result["all_lines"])
    assert result["total_traced_wall_us"]>0
    assert result["sum_line_intervals_us"]>0
    assert result["no_speedup_claim"] is True


def test_rejects_invalid_count_or_target_and_does_not_leave_tracer():
    for fn in (lambda:None,):
        with pytest.raises(ValueError):trace_lines(_arithmetic_probe,fn,0)
    with pytest.raises(ValueError):trace_lines(object(),lambda:None,4)
    assert sys.gettrace() is None


def test_must_actually_call_instrumented_target():
    with pytest.raises(ValueError,match="exactly requested"):
        trace_lines(_arithmetic_probe,lambda:None,3)
    assert sys.gettrace() is None


def test_original_transition_signature_stays_inherited():
    from scripts.langdecode057_validation import BatchedValidationTAC057
    assert BatchedValidationTAC057._advance is IncrementalTAC043._advance
    assert SAMPLES==24 and WARMUPS==2


def test_trace_measurements_are_only_instrumented_not_cpu_speedups():
    data=trace_lines(_arithmetic_probe,lambda:_arithmetic_probe(2),count=2)
    assert all(set(x)>={"line","source","samples","median_us","total_us"}
               for x in data["top_lines"])
    assert "speedup" not in data
    assert "production" not in data
