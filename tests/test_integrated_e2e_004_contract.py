from tac_osm.contract import load_contract
def test_e2e004_contract_is_consistent():
    c=load_contract("TACOSM-PLM-INTEGRATED-E2E-004")
    assert c.check_consistency()==[]
    assert c.primary_endpoint()=="heldout_q2_accuracy_typed_learned"
    assert {a.name for a in c.arms}=={"typed_learned","typed_oracle"}
