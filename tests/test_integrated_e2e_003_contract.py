from tac_osm.contract import load_contract


def test_e2e003_contract_is_consistent_and_has_one_primary():
    contract = load_contract("TACOSM-PLM-INTEGRATED-E2E-003")
    assert contract.check_consistency() == []
    assert contract.primary_endpoint() == "heldout_q2_accuracy_explicit_both"
    assert {arm.name for arm in contract.arms} == {
        "explicit_write", "explicit_read", "explicit_both"
    }

def test_e2e003_contract_excludes_invalid_predecessors():
    contract = load_contract("TACOSM-PLM-INTEGRATED-E2E-003")
    assert "E2E-001 and E2E-002 are invalid provenance-only artifacts" in " ".join(contract.held_constant)

def test_e2e003_post_action_update_uses_executed_action_outcome():
    from pathlib import Path
    source = (Path(__file__).resolve().parents[1] / "src" / "tac_osm" / "integrated_e2e.py").read_text()
    assert 'action1_hard = out1["logits"].argmax(dim=-1)' in source
    assert 'outcome1 = (action1_hard == q1[-1]).to(out1["logits"].dtype)' in source
    assert 'post_action_update(memory, out1, outcome1)' in source
    assert 'post_action_update(memory, out1, q1[-1])' not in source


def test_e2e003_address_model_inherits_correct_post_action_semantics():
    from pathlib import Path
    source = (Path(__file__).resolve().parents[1] / "src" / "tac_osm" / "integrated_e2e_address.py").read_text()
    assert 'post_action_update(memory, out1, outcome1)' in source
    assert 'post_action_update(memory, out1, q1[-1])' not in source
