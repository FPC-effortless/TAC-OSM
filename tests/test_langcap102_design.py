"""LANGCAP102 design policy tests: only constructed source lint, NEVER training."""
from importlib.util import module_from_spec,spec_from_file_location
from pathlib import Path
import json
import subprocess
import sys
import pytest

ROOT=Path(__file__).resolve().parents[1]
STUDY=ROOT/"research/langcap102"
spec=spec_from_file_location("langcap102_design_gate",STUDY/"gate.py")
gate=module_from_spec(spec)
spec.loader.exec_module(gate)

def study():
    return json.loads((STUDY/"PROTOCOL.json").read_text())

def policy():
    return (ROOT/"research/REAL-WORLD-EVIDENCE-RULES.md").read_text()

def test_design_only_and_approved_scale_without_training_authorization():
    r=gate.inspect_design(study(),policy())
    assert r["status"]=="PASS_NO_FIT_DESIGN_AND_APPROVED_SCALE_V2"
    assert r["training_authorized"] is False
    assert r["model_target_registered"] is False
    assert r["real_data_source_qualified"] is False

def test_reject_training_or_premature_model_size_registration():
    m=study()
    m["training_enabled"]=True
    with pytest.raises(gate.DesignGateError, match="training_enabled"):
        gate.inspect_design(m,policy())
    m=study()
    m["parameter_policy"]["preregistered_trainable_target"]=1_000_000
    with pytest.raises(gate.DesignGateError, match="preregistered"):
        gate.inspect_design(m,policy())

def test_fairness_and_genuine_source_requirements_cannot_be_removed():
    m=study()
    m["parameter_policy"]["between_arms_max_relative_gap"]=0.25
    with pytest.raises(gate.DesignGateError, match="1%"):
        gate.inspect_design(m,policy())
    m=study()
    m["datasets"][0]["status"]="ACCEPTED"
    with pytest.raises(gate.DesignGateError, match="unaudited"):
        gate.inspect_design(m,policy())
    m=study()
    m["primary_controls"].remove("faithful_decoder_only_transformer")
    with pytest.raises(gate.DesignGateError, match="Strong primary controls"):
        gate.inspect_design(m,policy())

def test_historical_original_failures_not_erased_by_amendment():
    m=study()
    m["prior_evidence"]["original_failures"]=0
    with pytest.raises(gate.DesignGateError, match="original record"):
        gate.inspect_design(m,policy())

def test_active_agent_instructions_use_registered_scale_and_preserve_history():
    agents=(ROOT/"AGENTS.md").read_text()
    assert "## Binding model-scale policy v2" in agents
    assert "declared-target-v2" in agents
    assert "LANGMATH101 and existing v1 studies retain 95,000–105,000" in agents
    assert "A standalone `--model-parameters` check is count-only" in agents
    assert "## Binding 100k trained-model scale" not in agents
    for name in ("CHARTER.md", "CODEX.md"):
        assert "scale-v2" in (ROOT/name).read_text()

def test_scientific_workflow_checks_current_policy_and_signed_evidence():
    text=(ROOT/".github/workflows/research-scientific-gate.yml").read_text()
    assert 'grep -q "Model-scale policy v2"' in text
    assert 'grep -q "Binding model-scale policy v2"' in text
    assert "python research/langcap102/gate.py design" in text
    assert "tests/test_reward_runner_integration.py" in text
    assert "--check-changed" in text
    assert "NEW_FITS_REQUIRE_100K" not in text
    assert "Compulsory prospective model-size budget" not in text

def test_readme_only_advertises_implemented_design_command():
    readme=(STUDY/"README.md").read_text()
    assert "gate.py readiness" not in readme
    assert "only implemented subcommand" in readme
    result=subprocess.run([sys.executable,str(STUDY/"gate.py"),"--help"],
                          capture_output=True,text=True,check=True)
    assert "{design}" in result.stdout
