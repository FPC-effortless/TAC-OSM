"""Tests for TACOSM-C5-ADMISSION-SCALING-AUDIT-001."""
from tac_osm.c5_admission_scaling_audit import (
    AuditConfig,
    build_population_extended,
    exact_best_valid_rank_from_class_scores,
    theoretical_l90,
    local_slopes,
)


def test_extended_population_supports_registered_high_M() -> None:
    candidates = build_population_extended(0, 8192)
    assert len(candidates) == 8192
    assert len({c.descriptor for c in candidates}) == 64
    counts = {c.descriptor: 0 for c in candidates}
    for c in candidates:
        counts[c.descriptor] += 1
    assert set(counts.values()) == {128}


def test_config_has_no_cap_intervention() -> None:
    cfg = AuditConfig()
    assert max(cfg.lsh_tables) == 64
    assert cfg.lsh_cap == 128


def test_theoretical_l90_matches_registered_formula_scale() -> None:
    l90 = theoretical_l90(0.86, 10)
    assert 8 <= l90 <= 10


def test_exact_rank_reducer_counts_repeated_class_copies() -> None:
    candidates = build_population_extended(3, 128)
    scores = [float(i) for i in range(64)]
    target = candidates[0].descriptor
    # Highest score class should have all 2 copies ahead only if target is not
    # that class; rank is therefore bounded by the repeated-class population.
    rank = exact_best_valid_rank_from_class_scores(candidates, target, scores)
    assert rank >= 1
    assert rank <= 128


def test_local_slope_matches_power_of_two_step() -> None:
    dense = [
        {"M": float(m), "P90_best_valid_rank": float(r)}
        for m, r in ((64, 2), (128, 3), (256, 5))
    ]
    slopes = local_slopes(dense)
    assert round(slopes[0]["local_P90_exponent"], 6) == round(__import__("math").log2(3 / 2), 6)
    assert round(slopes[1]["local_P90_exponent"], 6) == round(__import__("math").log2(5 / 3), 6)


def test_dense_extension_uses_same_noisy_trial_boundary() -> None:
    cfg = AuditConfig()
    assert cfg.dense_m_levels[-3:] == (2048, 4096, 8192)
    assert cfg.sweep_m == 1024
