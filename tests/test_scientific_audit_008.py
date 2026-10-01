from scripts.run_scientific_audit_008 import (
    deterministic_representability,
    make_unique_trial,
    pooled_quantile,
)
from tac_osm.c5_persistent_relational_loop import decode_state_value, OPS, DIM


def test_representability_is_deterministic_and_complete():
    a = deterministic_representability()
    b = deterministic_representability()
    assert a == b
    assert all(v["ranking_representable"] for v in a.values())
    assert all(v["witness_is_deterministic"] for v in a.values())


def test_pooled_quantile_is_trial_level():
    vals = [1, 1, 1, 100, 100]
    assert pooled_quantile(vals, 0.90) == 100


def test_structural_holdout_changes_support():
    train_trial, _ = make_unique_trial(0, 1, 64, mode="train")
    ood_trial, state = make_unique_trial(0, 2, 64, mode="ood")
    left, right, _ = decode_state_value(state.read(ood_trial.query).values[0])
    assert all(not (a == 1 and b == 1) for c in [train_trial] for bit in range(DIM)
               for a, b in [decode_state_value(state.read(ood_trial.query).values[0])[0][bit:bit+1][0],
                             decode_state_value(state.read(ood_trial.query).values[0])[1][bit:bit+1][0]])
    assert all(left[i] == 1 and right[i] == 1 for i in range(4))
    assert ood_trial.target_descriptor in {c.descriptor for c in ood_trial.candidates}
