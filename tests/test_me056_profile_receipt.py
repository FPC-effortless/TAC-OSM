import json
from pathlib import Path


def test_me056_receipt_passes_parameters_and_flops_but_blocks_cpu():
    receipt=json.loads(Path("research/me056-paired-execution-profile.json").read_text())
    assert receipt["model_fits_started"]==0
    assert receipt["parameter_relative_gap"]<=.01
    assert receipt["profiled_flop_relative_gap"]<=.01
    assert receipt["paired_median_bootstrap_95_interval"][0]>.05
    assert receipt["strict_5pct_gate_pass"] is False
    assert receipt["strict_5pct_gate_definitive_fail"] is True
    assert len(receipt["paired_TAC_over_GRU_relative_samples"])==41


def test_me056_receipt_contains_no_accuracy_or_capability_result():
    receipt=json.loads(Path("research/me056-paired-execution-profile.json").read_text())
    assert receipt["capability_claim"] is False
    assert not ({"MAE","MSE","policy_value","architecture_winner"}&set(receipt))
