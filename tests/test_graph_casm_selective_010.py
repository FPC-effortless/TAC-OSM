import importlib.util
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]
CASM_ROOT = ROOT / "third_party" / "cdl-attention-experiment"
if not CASM_ROOT.exists():
    pytest.skip("pinned external generator is not checked out", allow_module_level=True)

SPEC = importlib.util.spec_from_file_location("graph_casm_selective_010", ROOT / "scripts" / "run_graph_casm_selective_010.py")
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader
SPEC.loader.exec_module(MODULE)


def test_graph_feature_exposes_wiring_but_summary_does_not():
    ep = MODULE.generate(1001, 1)[0]
    graph = MODULE.features(ep, True)
    summary = MODULE.features(ep, False)
    assert len(graph) == MODULE.FEATURE_DIM
    assert len(summary) == MODULE.FEATURE_DIM
    assert graph[:90] == summary[:90]
    assert graph[90:] != summary[90:]


def test_exact_graph_executor_reproduces_complete_truth_table():
    ep = MODULE.generate(1002, 1)[0]
    check = MODULE.executor_check([ep])
    assert check["pass"]
    assert check["checked_rows"] == 16


def test_exact_graph_work_is_positive_and_proportional_to_rows():
    ep = MODULE.generate(1003, 1)[0]
    _, one = MODULE.execute_exact(ep, (0, 0, 0, 0))
    assert one.total > 0
    assert MODULE.candidate_work(ep, MODULE.VERIFIER_ROWS) == one.total * MODULE.VERIFIER_ROWS
