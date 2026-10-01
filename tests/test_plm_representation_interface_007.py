from scripts.run_plm_representation_interface_007 import representability_gate, effective_rank_pr

def test_representability_control_needs_constant_for_and_or():
    r=representability_gate()
    assert r["control"]["xor"] < 1e-8
    assert r["control"]["xnor"] < 1e-8
    assert r["control"]["and"] > 1e-8
    assert r["control"]["or"] > 1e-8
    assert all(v < 1e-8 for v in r["analytic"].values())

def test_effective_rank_is_bounded():
    assert 1.0 <= effective_rank_pr([(1,0),(0,1),(1,0),(0,1)]) <= 2.1
