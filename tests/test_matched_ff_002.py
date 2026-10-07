import json,inspect
import pytest
pytest.importorskip("torch")
from tac_osm.integrated_e2e_ff_feedback_baseline import FeedForwardFeedbackMultimodal
def test_parameter_match():
 m=FeedForwardFeedbackMultimodal();assert abs(m.parameter_count()-74126)/74126<=.01
def test_no_persistence():
 s=inspect.getsource(FeedForwardFeedbackMultimodal);assert "ExplicitEntityState" not in s and "FixedCASM" not in s and "post_action_update" not in s
def test_contract():
 c=json.load(open("contracts/TACOSM-PLM-MATCHED-FF-002.json"));assert c["status"]=="pre-registered" and c["primary_endpoint"]=="heldout_q2_accuracy"

# CI trigger only: contract and protocol unchanged.

def test_migrated_contract_schema_and_bound_history():
    from tac_osm.contract import load_contract

    for experiment_id, expected_arm in (
        ("TACOSM-PLM-MATCHED-FF-001", "FF-001"),
        ("TACOSM-PLM-MATCHED-FF-002", "FF-002"),
    ):
        c = load_contract(experiment_id)
        assert c.check_consistency() == []
        assert c.h_levels == (3,)
        assert c.seeds == (0, 1, 2, 3, 4)
        assert c.steps == 300
        assert c.eval_steps == 600
        assert [a.name for a in c.arms] == ["PLM", expected_arm]
        assert c.primary_endpoint() == "heldout_q2_accuracy"
        assert "A-schema-migration-001" in {a.id for a in c.amendments}

    ff2 = json.load(open("contracts/TACOSM-PLM-MATCHED-FF-002.json"))
    amendment = ff2["amendments"][0]
    assert amendment["affected"].startswith("schema/bookkeeping only")
    assert "66559497" in amendment["old_definition"]
    assert ff2["legacy_contract_schema"] == "flat-v0"
    assert ff2["schema_migration_note"].startswith("Bookkeeping-only migration")
