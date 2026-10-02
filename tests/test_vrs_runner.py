import importlib.util, json, subprocess, sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
SCRIPT=ROOT/"scripts/run_verified_representation_synthesis_001.py"
FIXTURE=ROOT/"research/vrs-001/proposal_smoke.json"

def load_runner():
    spec=importlib.util.spec_from_file_location("vrs_runner",SCRIPT)
    mod=importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod

def test_smoke_fixture_is_rejected_for_confirmatory():
    mod=load_runner()
    p=mod.Proposal.load(FIXTURE)
    try:
        p.require_confirmatory()
    except ValueError:
        pass
    else:
        raise AssertionError("smoke fixture must never be confirmatory")

def test_smoke_run_is_non_evidence():
    result=subprocess.run(
        [sys.executable,str(SCRIPT),"--proposal",str(FIXTURE),"--smoke"],
        cwd=ROOT,text=True,capture_output=True,check=True,
    )
    out=json.loads(result.stdout)
    assert out["status"]=="smoke"
    assert out["protocol"]["M"]==[64]
    assert out["protocol"]["K"]==[4,8]
    assert out["anchor_checks"] == 15

def test_scored_proposal_requires_query_coverage():
    mod=load_runner()
    raw=json.loads(FIXTURE.read_text())
    raw["mode"]="scored"
    raw["scored_states"]={"x":[0.0]}
    raw["scored_queries"]={}
    try:
        mod.Proposal(raw).audit()
    except ValueError:
        pass
    else:
        raise AssertionError("scored proposal must require scored query coverage")

def test_representation_distance_is_dimension_normalized():
    mod=load_runner()
    a=(0.0,)*5
    b=(0.35,)*5
    assert mod.representation_distance(a,b) == 0.35

def test_proposer_domain_excludes_private_evaluator_details():
    public_text=(ROOT/"research/vrs-001/PROPOSER-DOMAIN-001.md").read_text()
    private_text=(ROOT/"research/vrs-001/DOMAIN-001.md").read_text()
    assert "Exact task score" not in public_text
    assert "0.28" not in public_text
    assert "exact task score" in private_text.lower()
