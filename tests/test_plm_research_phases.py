from tac_osm.plm_research_phases import (
    operator_scaling_row,
    run_no_verifier_control,
    run_plasticity_sequence,
    run_state_stability,
)


def test_strict_verifier_blocks_all_wrong_writes():
    row = run_state_stability(seed=0, horizon=32, false_accept_rate=0.0)
    assert row.wrong_verified_writes == 0
    assert row.wrong_write_rate == 0.0


def test_false_acceptance_increases_contamination():
    row = run_state_stability(seed=0, horizon=32, false_accept_rate=1.0)
    assert row.wrong_verified_writes > 0
    assert row.wrong_write_rate > 0.0


def test_no_verifier_is_full_contamination_control():
    row = run_no_verifier_control(seed=0, horizon=32)
    assert row.wrong_verified_writes == row.wrong_attempts
    assert row.wrong_write_rate == 0.5


def test_operator_index_reports_raw_work_and_admission():
    row = operator_scaling_row(seed=1, e=1024)
    assert row.raw_candidates >= row.admitted_candidates
    assert row.indexed_cost_proxy >= row.probes + row.raw_candidates
    assert row.target_rank >= 1


def test_plasticity_has_heldout_metric():
    rows = run_plasticity_sequence(seed=0)
    assert len(rows) == 3
    assert all(0.0 <= row.heldout_accuracy <= 1.0 for row in rows)
