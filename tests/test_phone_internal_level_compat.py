"""Source-only planner integration. No neural execution or data admission."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from tac_osm.phone_research_ops import FAMILIES, ResearchPlanError, decide
from tac_osm.research_execution import validate_internal_research


def proposal(**changes):
    p = json.loads((ROOT / "research/phone_queue/phone_level_20261104.json").read_text())
    p.update(changes)
    return p


def assert_no_authority(d):
    assert d["approved_to_train"] is False
    assert d["approved_to_open_test"] is False
    assert d["automatic_submissions"] is False


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("kind", ["source_audit", "training", "evaluation"])
def test_internal_levels_parse_across_all_six_families_without_authority(family, kind):
    d = decide(proposal(variant=family, kind=kind, execution_level="internal_research"))
    assert d["execution_level"] == "internal_research"
    assert d["backend"] == "github_actions"
    assert not any("signed" in b or "independent evaluator" in b for b in d["blockers"])
    if kind != "source_audit":
        assert d["state"] == "BLOCKED_EVIDENCE_OR_DESIGN"
        assert any("source-authorized real data" in b for b in d["blockers"])
    assert_no_authority(d)


@pytest.mark.parametrize("kind", ["training", "evaluation"])
def test_legacy_default_matches_explicit_independent_qualification(kind):
    legacy = decide(proposal(kind=kind))
    explicit = decide(proposal(kind=kind, execution_level="independent_qualification"))
    assert legacy["execution_level"] == explicit["execution_level"] == "independent_qualification"
    assert legacy["blockers"] == explicit["blockers"]
    assert legacy["state"] == explicit["state"] == "BLOCKED_EVIDENCE_OR_DESIGN"
    assert any("signed" in b for b in legacy["blockers"])
    if kind == "evaluation":
        assert any("protected evaluation" in b for b in explicit["blockers"])
    assert_no_authority(legacy)


@pytest.mark.parametrize("value", [None, "force", "INTERNAL", True, [], {}])
def test_unknown_or_mistyped_levels_are_rejected(value):
    with pytest.raises(ValueError, match="execution level"):
        decide(proposal(execution_level=value))


def test_unknown_fields_remain_rejected():
    with pytest.raises(ResearchPlanError, match="exact proposal"):
        decide(proposal(execution_level="internal_research", grant_training=True))


def test_known_research_label_is_planning_only_even_with_fixture_metadata():
    d = decide(proposal(kind="training", execution_level="internal_research", seeds=[123],
                        real_source_sha256="a" * 64,
                        study_manifest="research/mathematical-engineering/fixture/FAIR_COMPARISON.json"))
    assert d["state"] == "READY_INTERNAL_RESEARCH_PLAN"
    assert_no_authority(d)  # Strings above are NOT an actual source receipt.


def test_internal_gpu_route_still_requires_two_devices_budget_and_workload():
    d = decide(proposal(execution_level="internal_research", requires_cuda=True))
    assert d["backend"] == "kaggle_t4x2" and d["gpu_devices_required"] == 2
    assert d["state"] == "BLOCKED_EVIDENCE_OR_DESIGN"
    assert any("budget" in b for b in d["blockers"])
    assert any("strategy" in b for b in d["blockers"])
    assert_no_authority(d)


def test_gru512_copy_parses_but_does_not_invent_a_real_source_or_release():
    path = ROOT / "research/phone_queue/local_binding_gru512_20261103.json"
    original = path.read_bytes()
    p = json.loads(original)
    assert "execution_level" not in p
    p["execution_level"] = "internal_research"
    d = decide(p)
    assert d["state"] == "BLOCKED_EVIDENCE_OR_DESIGN"
    assert d["blockers"] == ["source-authorized real data and preregistered scientific plan required"]
    assert path.read_bytes() == original
    assert_no_authority(d)
    status = json.loads((ROOT / "research/local_binding_gru512_20261103/DRAFT_STATUS.json").read_text())
    assert status["fitting_released"] is False
    assert status["root_RUN_REQUEST_present"] is False


def test_restored_helper_is_exact_existing_source_policy_not_a_new_admission_mode():
    vendored = ROOT / "research/collect012/policy/src/tac_osm/research_execution.py"
    assert (ROOT / "src/tac_osm/research_execution.py").read_bytes() == vendored.read_bytes()


@pytest.mark.parametrize("flag", ["protected_evaluation", "reward_integrity_certification",
                                  "safety_claim", "checkpoint_promotion"])
def test_restored_metadata_policy_retains_protected_scope_denial(flag):
    m = {"execution_level": "internal_research",
         "research_scope": {"publicly_licensed": True, "partitions": ["FIT"],
                            "protected_evaluation": False, "reward_integrity_certification": False,
                            "safety_claim": False, "checkpoint_promotion": False}}
    m["research_scope"][flag] = True
    with pytest.raises(ValueError, match=flag):
        validate_internal_research(m)


def test_no_torch_import_and_preserved_original_capsule_hashes():
    cmd = [sys.executable, "-B", "-c",
           "import sys; sys.path.insert(0,'src'); from tac_osm.phone_research_ops import decide; "
           "import json; decide(json.load(open('research/phone_queue/phone_level_20261104.json'))); "
           "assert 'torch' not in sys.modules"]
    subprocess.run(cmd, cwd=ROOT, check=True, capture_output=True, text=True)
    protocol = json.loads((ROOT / "research/phone_level_20261104/PROTOCOL.json").read_text())
    for path, digest in protocol["source_sha256"].items():
        if path != "src/tac_osm/phone_research_ops.py":
            assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
