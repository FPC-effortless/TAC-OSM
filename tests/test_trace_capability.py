from tac_osm.contract import load_contract


def test_016c_contract_is_single_primary():
    c = load_contract("TACOSM-GRAPH-CASM-TRACE-CAPABILITY-BRIDGE-016C-R1")
    assert c.check_consistency() == []
    assert c.primary_endpoint() == "verified_success_delta_m512_b8"


def test_budget_is_fixed_at_eight_in_contract():
    c = load_contract("TACOSM-GRAPH-CASM-TRACE-CAPABILITY-BRIDGE-016C-R1")
    assert c.sections["terminal_protocol"]["budget"] == 8


def test_scope_blocks_learned_claims():
    c = load_contract("TACOSM-GRAPH-CASM-TRACE-CAPABILITY-BRIDGE-016C-R1")
    assert c.sections["scope"]["learned_probe_policy_claim"] is False


def test_budget_capped_utility_matches_partition_ceiling():
    from tac_osm.trace_capability import budget_capped_utility
    evidence = (0, 0, 1, 1)
    assert budget_capped_utility(evidence, 1) == 0.5
    assert budget_capped_utility(evidence, 2) == 1.0


def test_budget_capped_utility_fails_closed():
    from tac_osm.trace_capability import budget_capped_utility
    import pytest
    with pytest.raises(ValueError):
        budget_capped_utility((), 8)
    with pytest.raises(ValueError):
        budget_capped_utility((0, 1), 0)


def test_scalar_target_evidence_signature_matches_candidate_signature():
    candidate_evidence = ((0,), (1,), (0,))
    target_evidence = (0,)
    assert all(isinstance(x, tuple) for x in candidate_evidence)
    assert isinstance(target_evidence, tuple)
    assert sum(sig == target_evidence for sig in candidate_evidence) == 2
