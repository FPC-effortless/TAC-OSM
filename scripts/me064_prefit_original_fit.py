"""ME064 task-stratified real FIT preflight. ZERO optimizer steps.

Checks source hashes, 5 task-balanced factual original FIT batches, all
gradients, and supported-operator FLOPs. No CAL or unvisited target labels.
"""
import argparse, json, platform
from pathlib import Path
import torch

from scripts.me064_mixed_real_fit import fit_metadata, batch, SEEDS
from scripts.me063_prefit_original_fit import model_for, profile_once, CONFIG
from scripts.me060_train_trajectory import digest, BATCH, HISTORY, HORIZON, UPDATES


def preflight(cache,output):
    torch.set_num_threads(1)
    task_plan,identified,source,metadata,orders=fit_metadata(cache)
    real_minibatch=batch(source,orders[str(SEEDS[0])][0])
    comparison={}
    for arch in CONFIG:
        torch.manual_seed(SEEDS[0])
        comparison[arch]=profile_once(model_for(arch,identified),real_minibatch)
    counts=[r["parameters"] for r in comparison.values()]
    flops=[r["profiled_forward_backward_flops"] for r in comparison.values()]
    metadata.update({
        "registered_seeds":list(SEEDS),
        "per_architecture":comparison,
        "parameter_relative_gap":(max(counts)-min(counts))/max(counts),
        "supported_operator_flop_relative_gap":(max(flops)-min(flops))/max(flops),
        "registered_budget":{"updates":UPDATES,"batch":BATCH,"history":HISTORY,
                             "horizon":HORIZON,"per_task_updates":UPDATES//5},
        "full_hardware_flop_parity_certified":False,
        "cpu_budget_parity_required":False,
        "training_status":"NOT_RUN",
        "scientific_status":"ONLY_REAL_FIT_RESOURCE_MEASUREMENT_NOT_MODEL_SKILL",
        "calibration_outcomes_opened":False,"unseen_target_outcomes_opened":False,
        "python":platform.python_version(),"torch":torch.__version__,
        "original_target_task":task_plan["holdout_task"],
    })
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(metadata,sort_keys=True,indent=2)+"\n")
    print("ME064_AUTHENTIC_MULTITASK_PREFLIGHT="+json.dumps({
        "fit_original_sha256":metadata["fit_original_sha256"],
        "per_architecture":{k:{"parameters":v["parameters"],
             "profiled_flops":v["profiled_forward_backward_flops"]} for k,v in comparison.items()},
        "param_gap":metadata["parameter_relative_gap"],
        "flop_gap":metadata["supported_operator_flop_relative_gap"],
        "optimizer_updates":0,"holdout_outcomes_opened":False
        }),flush=True)
    if metadata["parameter_relative_gap"]>.01 or metadata["supported_operator_flop_relative_gap"]>.01:
        raise SystemExit("ME064_PRETRAINING_RESOURCE_PARITY_FAIL")
    if not all(all(v["has_finite_gradients"].values()) for v in comparison.values()):
        raise SystemExit("ME064_MISSING_ACTIVE_PARAMETER_GRADIENTS")
    print("ME064_REAL_MULTITASK_PREFIT_PARITY_PASS_NO_TRAINING",flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--cache",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    a=parser.parse_args()
    preflight(a.cache,a.output)
