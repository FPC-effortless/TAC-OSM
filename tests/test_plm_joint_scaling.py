from scripts.run_plm_joint_scaling_005 import evaluate


def test_joint_row_charges_both_addressing_surfaces():
    row = evaluate(seed=0, m=64, e=64)
    assert row.address_cost >= row.state_raw_candidates + row.operator_raw_candidates
    assert row.total_cost_proxy == row.address_cost + row.execution_cost


def test_joint_row_keeps_independent_population_axes():
    row = evaluate(seed=0, m=64, e=256)
    assert row.M == 64
    assert row.E == 256
    assert row.full_scan_proxy == 320


def test_joint_success_is_stricter_than_target_admission():
    row = evaluate(seed=0, m=1024, e=1024)
    assert row.joint_success <= (
        row.state_target_admitted and row.operator_target_admitted
    )
