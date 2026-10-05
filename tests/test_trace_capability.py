from tac_osm.contract import load_contract


def test_016c_contract_is_single_primary():
    c = load_contract("TACOSM-GRAPH-CASM-TRACE-CAPABILITY-BRIDGE-016C")
    assert c.check_consistency() == []
    assert c.primary_endpoint() == "verified_success_delta_m512_b8"


def test_budget_is_fixed_at_eight_in_contract():
    c = load_contract("TACOSM-GRAPH-CASM-TRACE-CAPABILITY-BRIDGE-016C")
    assert c.sections["terminal_protocol"]["budget"] == 8


def test_scope_blocks_learned_claims():
    c = load_contract("TACOSM-GRAPH-CASM-TRACE-CAPABILITY-BRIDGE-016C")
    assert c.sections["scope"]["learned_probe_policy_claim"] is False
