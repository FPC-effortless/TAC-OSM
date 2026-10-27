"""Constructed falsifiers for scoring; no scientific performance evidence."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from tac_osm.historical_query_metrics import window_scores, station_gain_ci, compare_errors

ROOT = Path(__file__).resolve().parents[1]


def test_perfect_content_and_wrong_timestamp_control():
    y = list(map(float, range(8)))
    s = window_scores(y, y, [True] * 8)
    assert s["H64_abs_error"] == 0
    assert s["centered_MSE"] == 0
    assert s["centered_null_MSE"] > 0
    assert s["circular_wrong_query_MAE"] > 0
    assert s["H64_wrong_clock_mean_abs_error"] > 0


def test_episode_level_predictor_has_no_centered_skill_or_query_specificity():
    y = [100. + i for i in range(8)]
    p = [103.5] * 8
    s = window_scores(p, y, [True] * 8)
    assert s["prediction_range"] == 0
    assert s["centered_MSE"] == s["centered_null_MSE"]
    assert s["H64_abs_error"] == s["H64_wrong_clock_mean_abs_error"]
    assert s["all_query_MAE"] == s["collapsed_query_MAE"]


def test_target_missingness_never_becomes_zero_or_pass():
    s = window_scores([1.] * 8, [2.] + [None] * 7, [True] + [False] * 7)
    assert s["observed_targets"] == 1
    assert s["centered_MSE"] is None
    assert s["centered_status"].startswith("INCONCLUSIVE")
    with pytest.raises(ValueError):
        window_scores([1.] * 8, [None] * 8, [True] * 8)


def test_scoring_centering_removes_between_window_level_only():
    y = [float(i) for i in range(8)]
    p = [i * .7 for i in y]
    a = window_scores(p, y, [True] * 8)
    b = window_scores([v + 50 for v in p], [v + 100 for v in y], [True] * 8)
    assert a["centered_MSE"] == pytest.approx(b["centered_MSE"])
    assert a["centered_null_MSE"] == pytest.approx(b["centered_null_MSE"])


def test_station_gain_keeps_source_units_and_does_not_promote_ties():
    sites = [0, 0, 1, 1, 2, 2]
    assert station_gain_ci(sites, [2.] * 6, [1.] * 6, draws=100) == [1., 1.]
    assert compare_errors(sites, [1.] * 6, [1.] * 6)["gate_pass"] is False
    with pytest.raises(ValueError):
        station_gain_ci(sites, [1.], [0.], draws=100)


def test_preflight_is_internal_cpu_and_contains_no_source_evaluation(tmp_path):
    out = tmp_path / "preflight.json"
    r = subprocess.run([sys.executable, str(ROOT / "scripts/audit_content_query_specificity.py"),
                        "--output", str(out)], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    report = json.loads(out.read_text())
    assert report["status"] == "READY_INTERNAL_RESEARCH"
    assert report["evaluation_completed"] is False
    assert report["scientific_qualification"] is False
    assert report["compute_decision"]["backend"] == "github_actions"


def test_original_content_graph_uses_legal_window_and_query_shapes():
    original = os.environ.get("TAC_OSM_CONTENT_QUERY_SOURCE")
    if not original:
        pytest.skip("dedicated CPU workflow provides original content-model code")
    code = r'''
from pathlib import Path
import sys
sys.path[:0] = [str(Path.cwd()), str(Path.cwd()/"src")]
from scripts.original_history_trace import verify_implementation, load_original_modules
from scripts.audit_content_query_specificity import predict_batch
import torch
root = verify_implementation(Path(sys.argv[1]), "fd201207cb76b30470f64e1fc1baecf9de44560d")
source, original = load_original_modules(root)
torch.set_num_threads(1)
torch.manual_seed(1)
windows = dict(values=torch.randn(2,88,13),valid=torch.ones(2,88,13,dtype=torch.bool),
               age=torch.zeros(2,88,13),last=torch.zeros(2,88,13))
owners = torch.tensor([0,1])
clocks = source.calendar(list(range(0,64,8))*2).reshape(2,8,6)
with torch.no_grad():
 for arm in ("assoc_historic_query","fast_slow_historic_query"):
  m = original.create_model(1,arm).eval()
  p,z,err = predict_batch(m,windows,owners,clocks)
  assert p.shape == z.shape == (2,8)
  assert err <= 2e-6
  if arm == "fast_slow_historic_query":
   assert torch.equal(p,z), "matrix-only lesion must not affect the actual nonassociative historical graph"
  try:
   predict_batch(m,windows | {"raw_targets":torch.zeros(2,8)},owners,clocks)
  except ValueError:
   pass
  else:
   raise AssertionError("denied raw target field accepted")
print("PASS_ORIGINAL_CODE_FIXTURE_ONLY")
'''
    r = subprocess.run([sys.executable, "-c", code, original], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "PASS_ORIGINAL_CODE_FIXTURE_ONLY" in r.stdout
