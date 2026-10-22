import json
from pathlib import Path


def test_authentic_prefit_receipt_blocks_training_on_strict_cpu_gate():
    receipt=json.loads(Path("research/me055-prefit-profile.json").read_text())
    assert receipt["study"]=="ME055_PREFIT_AUTHENTIC_RESOURCE_PROFILE"
    assert receipt["model_fits_started"]==0
    assert receipt["strict_parameter_gate_pass"] is True
    assert receipt["strict_FLOP_gate_pass"] is True
    assert receipt["strict_CPU_gate_pass"] is False
    assert receipt["parameter_relative_gap"]<=.01
    assert receipt["profiled_FLOP_relative_gap"]<=.01
    assert receipt["measured_CPU_relative_gap"]>.05
    assert {row["architecture"] for row in receipt["arms"]}=={
        "native_tac","gru_world"}


def test_no_capability_result_is_encoded_in_prefit_receipt():
    receipt=json.loads(Path("research/me055-prefit-profile.json").read_text())
    forbidden={"test_MAE","dev_MAE","policy_value","accuracy",
               "architecture_winner","capability_gate_pass"}
    assert forbidden.isdisjoint(receipt)
