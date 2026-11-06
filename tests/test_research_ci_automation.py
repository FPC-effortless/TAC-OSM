"""Prevent regression from automatic, fail-closed research CI to manual-only gates.

These tests verify orchestration wiring, not a model's scientific capability.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = ROOT / ".github" / "workflows"


def workflow(name):
    return (WORKFLOWS / name).read_text(encoding="utf-8")


def test_both_research_gate_workflows_run_on_every_pr_and_mainline_push():
    for name in ("research-anticheat.yml", "research-observatory.yml"):
        text = workflow(name)
        assert "on:\n  pull_request:\n    branches: [main, master]" in text
        assert "  push:\n    branches: [main, master]" in text
        assert "  workflow_dispatch:" in text
        assert "    paths:" not in text, "path filters can accidentally skip required gates"


def test_anticheat_gate_runs_adversarial_tests_and_fixture_only_contract():
    text = workflow("research-anticheat.yml")
    assert "tests/test_research_anticheat.py tests/test_robot_anticheat.py" in text
    assert "scripts/run_robot_anticheat.py --fixture" in text
    assert 'report["source_provenance"] == "constructed_fixture_not_capability"' in text
    assert "junitxml=ci-artifacts/anticheat-tests.xml" in text


def test_observatory_gate_runs_tomography_and_batched_isolation_tests():
    text = workflow("research-observatory.yml")
    assert "tests/test_research_observatory.py" in text
    assert "--fixture --entity-isolation --profile" in text
    assert 'r["gradient_conflicts"]["status"] == "NOT_RUN_NO_AUTHENTIC_SUPERVISED_LOSSES"' in text
    assert "junitxml=ci-artifacts/observatory-tests.xml" in text


def test_ci_artifacts_are_explicitly_no_fit_and_workflows_have_read_only_permission():
    for name in ("research-anticheat.yml", "research-observatory.yml"):
        text = workflow(name)
        assert "permissions:\n  contents: read" in text
        assert "if: always()" in text
        assert "uses: actions/upload-artifact@v4" in text
        assert "retention-days: 14" in text
        assert "retention-days: 14" in text
        assert "--fixture" in text



def test_reward_hacking_negative_controls_are_compulsory_in_automatic_anticheat_ci():
    text = workflow("research-anticheat.yml")
    assert "tests/test_research_reward_integrity.py tests/test_run_reward_integrity.py" in text
    assert "scripts/run_reward_integrity.py --fixture" in text
    assert "ci-artifacts/reward-hacking-negative-control.json" in text
    assert "assert r['hacked_case_status'] == 'FAIL'" in text
    assert "assert r['reward_promotion_status'] == 'FAIL'" in text



def test_signed_reward_runner_attestations_remain_mandatory_in_ci():
    anticheat = workflow("research-anticheat.yml")
    science = workflow("research-scientific-gate.yml")
    assert "tests/test_reward_runner_integration.py" in anticheat
    assert "tests/test_reward_runner_integration.py" in science
    assert "cryptography" in anticheat
    assert "cryptography" in science
    parity_source = (ROOT / "scripts" / "validate_research_parity.py").read_text()
    assert "verify_reward_attestation" in parity_source
    assert "TAC_OSM_REWARD_VERIFIER_PUBLIC_KEY_HEX" in parity_source
    assert "scripts/run_qualified_research.py" in parity_source
