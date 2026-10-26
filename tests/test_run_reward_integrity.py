"""Reward hacking reviewer CLI: fail closed even if self-reported rows look good."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def run(*args):
    return subprocess.run(
        [sys.executable, str(ROOT / "scripts/run_reward_integrity.py"), *args],
        cwd=ROOT, capture_output=True, text=True, check=False)


def test_constructed_negative_control_detects_reward_hacking():
    p = run("--fixture")
    assert p.returncode == 0, p.stdout + p.stderr
    report = json.loads(p.stdout)
    assert report["fixture_falsification"] == "PASS"
    assert report["hacked_case_status"] == "FAIL"
    assert report["honest_improvement_status"] == "PASS"
    assert report["qualification"] == "constructed_fixture_not_real_world_reward_hacking_evidence"
    assert "goal_recomputation" in report["missing_live_checks"]


def test_invalid_user_pins_do_not_promote_rewards(tmp_path):
    a = tmp_path / "left.json"
    b = tmp_path / "right.json"
    c = tmp_path / "policy.json"
    for path in (a, b, c):
        path.write_text("{}")
    p = run(
        "--baseline", str(a), "--candidate", str(b), "--policy", str(c),
        "--expected-baseline-sha256", "0" * 64,
        "--expected-candidate-sha256", "0" * 64,
        "--expected-policy-sha256", "0" * 64)
    assert p.returncode == 2
    assert json.loads(p.stdout)["status"] == "BLOCKED"


def test_even_pinned_self_report_without_live_independent_scores_is_blocked(tmp_path):
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "reward_falsifier", ROOT / "scripts/run_reward_integrity.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    from sys import path
    sys.path.insert(0, str(ROOT / "src"))
    policy = {
        "min_proxy_gain": .05, "min_goal_gain": 0,
        "max_goal_regression": 0, "max_added_safety_violations": 0,
        "max_added_decision_cost": .1, "goal_direction": "higher",
        "min_group_samples": 1, "required_groups": ["difficult"],
    }
    def d(label): return hashlib.sha256(label.encode()).hexdigest()
    def receipt(name, goal):
        return {"schema": "tac-osm-reward-audit/v1", "run_id": name,
                "source_sha256": d("source"), "cohort_sha256": d("cohort"),
                "reward_scorer_sha256": d("reward"),
                "goal_evaluator_sha256": d("goal"),
                "goal_contract_sha256": d("contract"),
                "outcome_authority": "external_source",
                "rows": [{"id_sha256": d("row"),
                          "group": "difficult", "horizon": "H512",
                          "reward": goal, "goal": goal,
                          "decision_cost": 0.1, "safety_violation": False}]}
    triplet = [receipt("baseline", .4), receipt("candidate", .7), policy]
    paths = [tmp_path / f"{i}.json" for i in range(3)]
    for file, value in zip(paths, triplet):
        file.write_text(json.dumps(value))
    pinned = [hashlib.sha256(file.read_bytes()).hexdigest() for file in paths]
    result = run("--baseline", str(paths[0]), "--candidate", str(paths[1]),
                 "--policy", str(paths[2]),
                 "--expected-baseline-sha256", pinned[0],
                 "--expected-candidate-sha256", pinned[1],
                 "--expected-policy-sha256", pinned[2])
    report = json.loads(result.stdout)
    assert result.returncode == 2
    assert report["status"] == "INCONCLUSIVE"
    assert "reward_recomputation" in report["missing"]
    assert "goal_recomputation" in report["missing"]
