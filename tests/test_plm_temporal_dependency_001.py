from tac_osm.contract import load_contract
from tac_osm.temporal_dependency_001_benchmark import (
    Q1_QUERY,
    Q2_QUERY,
    sample_episode,
    sample_evaluation_episodes,
    validate_episode,
)
import random


def test_temporal_dependency_contract():
    c = load_contract("TACOSM-PLM-TEMPORAL-DEPENDENCY-001")
    assert c.check_consistency() == []
    assert c.seeds == (0, 1, 2, 3, 4)
    assert c.steps == 300
    assert c.eval_steps == 600


def test_temporal_benchmark_masks_only_q1_secret():
    ep = sample_episode(random.Random(1234), secret=1, visible_q2_bit=0)
    validate_episode(ep)
    pre, post, q1, q2, payload = ep
    assert tuple(q1[:4]) == Q1_QUERY
    assert tuple(q2[:4]) == Q2_QUERY
    assert int(pre[0][1][4]) == 21
    assert int(post[0][1][4]) == 20
    assert payload[2] == 1
    assert payload[3] == 0


def test_temporal_eval_balance_is_exact():
    episodes = sample_evaluation_episodes(random.Random(42), 600)
    cells = {}
    targets = []
    for ep in episodes:
        secret = ep[4][2]
        visible = ep[4][3]
        cells[(secret, visible)] = cells.get((secret, visible), 0) + 1
        targets.append(ep[3][4])
    assert cells == {(0, 0): 150, (1, 0): 150, (0, 1): 150, (1, 1): 150}
    assert sum(targets) == 300
