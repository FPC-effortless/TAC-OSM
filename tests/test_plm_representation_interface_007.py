from scripts.run_plm_representation_interface_007 import representability_gate, effective_rank_pr

def test_representability_gate_checks_ranking_capability():
    r=representability_gate()
    assert all(v["ranking_representable"] for v in r["control"].values())
    assert all(v["ranking_representable"] for v in r["analytic"].values())
    assert r["control"]["and"]["exact_signed_output_requires_constant"]
    assert r["control"]["or"]["exact_signed_output_requires_constant"]

def test_effective_rank_is_bounded():
    assert 1.0 <= effective_rank_pr([(1,0),(0,1),(1,0),(0,1)]) <= 2.1
