"""Internal measurement cannot silently become independent qualification."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from scripts.audit_original_history import preflight
from scripts.validate_research_parity import check_manifest, ParityError
from tac_osm.research_execution import execution_level, validate_internal_research

ROOT = Path(__file__).resolve().parents[1]


def plan():
    return json.loads((ROOT / "research/mathematical-engineering/history_audit014/FAIR_COMPARISON.json").read_text())


def test_ordinary_public_fit_research_needs_no_signers(monkeypatch):
    for name in ("TAC_OSM_SOURCE_VERIFIER_PUBLIC_KEY_HEX", "TAC_OSM_EVALUATOR_VERIFIER_PUBLIC_KEY_HEX"):
        monkeypatch.delenv(name, raising=False)
    _, report = preflight()
    assert report["status"] == "READY_INTERNAL_RESEARCH"
    assert report["scientific_qualification"] is False
    assert report["evaluation_executed"] is False
    assert report["source_receipt"]["license"] == "CC-BY-4.0"
    result = check_manifest(plan(), "preflight")
    assert result["reward_integrity"] == "NOT_INDEPENDENTLY_CERTIFIED_INTERNAL_RESEARCH"
    assert result["eligible_for_architecture_superiority"] is False


@pytest.mark.parametrize("flag", ["protected_evaluation", "reward_integrity_certification",
                                  "safety_claim", "checkpoint_promotion"])
def test_internal_label_cannot_bypass_qualification(flag):
    m = plan()
    m["research_scope"][flag] = True
    with pytest.raises(ValueError, match="independent qualification"):
        validate_internal_research(m)
    with pytest.raises(ParityError):
        check_manifest(m, "preflight")


@pytest.mark.parametrize("fault", ["TEST", "license", "hash", "split", "causality", "checkpoint"])
def test_internal_still_requires_source_and_reproducibility(fault):
    m = plan()
    if fault == "TEST":
        m["research_scope"]["partitions"] = ["TEST"]
    elif fault == "license":
        m["real_world_source"]["license"] = ""
    elif fault == "hash":
        m["real_world_source"]["sha256"] = "unknown"
    elif fault == "split":
        m["split_policy"]["frozen"] = False
    elif fault == "causality":
        m["causal_protocol"] = ""
    else:
        m["checkpoint_policy"] = ""
    with pytest.raises(ValueError):
        validate_internal_research(m)


def test_internal_level_does_not_skip_parameter_parity():
    m = plan()
    m.update(study_type="single_model_training", registered_before_first_fit=True,
        models=[dict(variant="fixture", seed=1, trainable_parameters=50000, deployed_parameters=50000)])
    with pytest.raises(ParityError):
        check_manifest(m, "preflight")


def test_default_remains_independent_and_unknown_levels_fail():
    assert execution_level({}) == "independent_qualification"
    with pytest.raises(ValueError):
        execution_level({"execution_level": "force"})
