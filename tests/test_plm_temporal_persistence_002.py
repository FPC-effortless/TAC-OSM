import random

from tac_osm.temporal_persistence_002_benchmark import (
    Q1_QUERY,
    Q2_QUERY,
    SECRET_BIT,
    VISIBLE_Q2_BIT,
    sample_episode,
    sample_evaluation_episodes,
    validate_episode,
)


def test_pure_persistence_contract_and_surface():
    from tac_osm.contract import load_contract
    c = load_contract("TACOSM-PLM-TEMPORAL-PERSISTENCE-002")
    assert c.check_consistency() == []
    assert c.seeds == (0, 1, 2, 3, 4)
    assert c.steps == 300
    assert c.eval_steps == 600


def test_q1_and_q2_use_distinct_entities():
    ep = sample_episode(random.Random(123), secret=1, visible_q2_bit=0)
    validate_episode(ep)
    pre, post, q1, q2, payload = ep
    assert q1[:4] == Q1_QUERY
    assert q2[:4] == Q2_QUERY
    assert q1[0] != q2[0]
    assert int(pre[1][1][4]) == 21
    assert int(post[1][4]) == 20
    assert payload[q2[0]][SECRET_BIT] == 1
    assert payload[q2[0]][VISIBLE_Q2_BIT] == 0


def test_evaluation_balance():
    episodes = sample_evaluation_episodes(random.Random(42), 600)
    cells = {}
    for ep in episodes:
        payload = ep[4]
        cell = (payload[1][SECRET_BIT], payload[1][VISIBLE_Q2_BIT])
        cells[cell] = cells.get(cell, 0) + 1
    assert cells == {(0,0):150,(1,0):150,(0,1):150,(1,1):150}
