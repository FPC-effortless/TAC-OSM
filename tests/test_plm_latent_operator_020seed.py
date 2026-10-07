import json

from tac_osm.contract import load_contract


def test_c19_20seed_extension_contract():
    c = load_contract("TACOSM-PLM-LATENT-OPERATOR-020SEED")
    assert c.check_consistency() == []
    assert c.seeds == tuple(range(20))
    assert c.steps == 300
    assert c.eval_steps == 600
