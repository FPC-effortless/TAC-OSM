"""Real-window no-fit anticheat enforcement; fixture-only tests."""
from dataclasses import replace
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

import pytest

from tac_osm.robot_anticheat import audit_robot_windows
from tac_osm.real_robot_transition_contract import RealLoggedWindow
from tac_osm.research_anticheat import AuditGateError

ROOT = Path(__file__).resolve().parents[1]


def window(frame=1):
    v = (0.0,) * 14
    return RealLoggedWindow((v, v), (v,), v, v, False,
                            (1.0,) * 14, False, frame, 7)


def test_empty_robot_dataset_fails_closed():
    ledger = audit_robot_windows([], run_id="none")
    assert ledger.report()["status"] == "FAIL"
    with pytest.raises(AuditGateError):
        ledger.assert_clean()


def test_original_window_contract_gates_clean():
    ledger = audit_robot_windows([window(1), window(2)], run_id="a")
    assert ledger.report()["status"] == "PASS"
    ledger.assert_clean()


def test_bad_robot_history_is_localized():
    s = replace(window(), currently_human_controlled=True,
                new_intervention_onset=True)
    r = audit_robot_windows([s], run_id="wrong")
    assert r.report()["status"] == "FAIL"
    assert "invalid causal window" in str(r.report())


def test_temporally_reversed_episodes_fail():
    assert audit_robot_windows([window(2), window(1)],
                               run_id="reverse").report()["status"] == "FAIL"


def test_robot_fixture_cli_writes_explicit_noncapability_status():
    p = subprocess.run(
        [sys.executable, str(ROOT / "scripts/run_robot_anticheat.py"), "--fixture"],
        capture_output=True, text=True, cwd=ROOT, check=False)
    assert p.returncode == 0, p.stderr
    data = json.loads(p.stdout)
    assert data["status"] == "PASS"
    assert data["source_provenance"] == "constructed_fixture_not_capability"


def test_robot_source_cli_rejects_unpinned_bytes(tmp_path):
    p = tmp_path / "window.jsonl"
    p.write_text('{"frame": 1}\n', encoding="utf-8")
    run = subprocess.run(
        [sys.executable, str(ROOT / "scripts/run_robot_anticheat.py"),
         "--windows-jsonl", str(p), "--expected-sha256", "0" * 64],
        capture_output=True, text=True, cwd=ROOT, check=False)
    assert run.returncode == 2
    assert json.loads(run.stdout)["status"] == "FAIL"
