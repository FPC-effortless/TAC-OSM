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
