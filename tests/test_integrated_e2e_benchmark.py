import random

from tac_osm.integrated_e2e_benchmark import (
    HELDOUT,
    TRAIN_COMBOS,
    sample_episode,
    validate_episode,
)


def test_corrected_generator_targets_distinct_query_entities():
    rng = random.Random(12345)
    for _ in range(200):
        ep = sample_episode(rng)
        assert ep[1][0] != ep[2][0]
        assert ep[1][:3] in TRAIN_COMBOS
        assert ep[2][:3] in TRAIN_COMBOS


def test_every_heldout_composition_is_excluded_from_training():
    assert all(combo not in TRAIN_COMBOS for combo in HELDOUT)


def test_corrected_generator_handles_each_registered_heldout_pair():
    rng = random.Random(20261002)
    for combo1 in HELDOUT:
        for combo2 in HELDOUT:
            if combo1 == combo2:
                continue
            ep = sample_episode(rng, combo1=combo1, combo2=combo2)
            validate_episode(ep)
            assert ep[1][:3] == combo1
            assert ep[2][:3] == combo2
            assert ep[1][0] != ep[2][0]
