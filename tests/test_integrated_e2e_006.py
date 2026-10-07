import random

from tac_osm.integrated_e2e_006_benchmark import (
    ALL_COMBOS,
    DEV_COMBOS,
    HELDOUT,
    SEALED_E2E005,
    TRAIN_COMBOS,
    sample_evaluation_episodes,
    validate_episode,
)

def modality(bit):
    return 0 if bit < 4 else 1 if bit < 8 else 2

def test_e2e006_split_is_disjoint_and_complete():
    assert set(HELDOUT).isdisjoint(SEALED_E2E005)
    assert set(HELDOUT).isdisjoint(DEV_COMBOS)
    assert set(HELDOUT).isdisjoint(TRAIN_COMBOS)
    assert set(DEV_COMBOS).isdisjoint(TRAIN_COMBOS)
    assert set(SEALED_E2E005).isdisjoint(TRAIN_COMBOS)
    assert len(HELDOUT) == 8
    assert len(DEV_COMBOS) == 32
    assert len(TRAIN_COMBOS) + len(HELDOUT) + len(DEV_COMBOS) + len(SEALED_E2E005) == len(ALL_COMBOS)

def test_e2e006_heldout_covers_cross_modal_pairs():
    assert all(modality(i) != modality(j) for _, i, j in HELDOUT)

def test_e2e006_heldout_sampler_is_registered_and_label_correct():
    episodes = sample_evaluation_episodes(random.Random(606006), 64)
    for ep in episodes:
        validate_episode(ep)
