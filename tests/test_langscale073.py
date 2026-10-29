"""LANGSCALE073 constructed fail-closed replay interval; no fit or DEV metrics."""
import json
from pathlib import Path
import tempfile
import pytest
from scripts.langscale073_aggregate import aggregate,SEEDS,TIERS,TARGET_SHA

def fixture_row(tier,seed):
    identity="b"*64
    raw={"context":128,"n":256,"nll":2.1,"brier":.75,"top1":.4,
         "inputs_and_targets_sha256":identity}
    r256={**raw,"context":256,"nll":2.3,"inputs_and_targets_sha256":"c"*64}
    cap256={**r256,"nll":2.17}
    return {"study":"LANGSCALE073","tier":tier,"seed":seed,"training_updates":0,
            "gpu_minutes":0,"protected_TEST":"NOT_ACCESSED",
            "target_cohort_sha256":TARGET_SHA,"same_targets":256,
            "hook_removed_and_model_weights_original":True,
            "cross_host_replay_abs_tolerance":1e-6,
            "prior_LANGSCALE072_exact_replay_status":"FAILED_4_OF_9",
            "cross_host_replay_errors":{str(k):{"abs_nll":2e-7,
                    "abs_brier":2e-7,"top1_identical":True,
                    "source_input_sha_identical":True} for k in (128,256)},
            "original_position_128":raw,
            "cap_position_128_negative_control":dict(raw),
            "original_position_256":r256,
            "cap_position_256":cap256,
            "cap256_minus_native256_nll":cap256["nll"]-r256["nll"]}

def make_rows(tmp,perturb=None):
    for tier in TIERS:
        for seed in SEEDS:
            row=fixture_row(tier,seed)
            if perturb and (tier,seed)==("p250k",1701):
                perturb(row)
            d=tmp/f"{tier}-seed{seed}"
            d.mkdir()
            (d/"POSITION_RESULTS.json").write_text(json.dumps(row))

def test_all_nine_complete_frozen_controls(tmp_path):
    make_rows(tmp_path)
    result=aggregate(tmp_path)
    assert result["model_source_checkpoints"]==9
    assert result["prior_LANGSCALE072_exact_replay_status"]=="FAILED_4_OF_9"
    assert result["cross_host_replay_abs_tolerance"]==1e-6
    assert result["metrics"]["p500k"]["cap_improvement_count"]==3

@pytest.mark.parametrize("mutation",[
    lambda r:r["cross_host_replay_errors"]["128"].update(abs_nll=1.01e-6),
    lambda r:r["cross_host_replay_errors"]["256"].update(abs_brier=1.01e-6),
    lambda r:r["cross_host_replay_errors"]["128"].update(source_input_sha_identical=False),
    lambda r:r["cross_host_replay_errors"]["128"].update(top1_identical=False),
    lambda r:r.update(prior_LANGSCALE072_exact_replay_status="PASS"),
    lambda r:r["cap_position_128_negative_control"].update(nll=3.1)
])
def test_fail_closed_drift_source_and_negative_controls(tmp_path,mutation):
    make_rows(tmp_path,mutation)
    with pytest.raises(ValueError):
        aggregate(tmp_path)
