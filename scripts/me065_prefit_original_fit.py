"""Source-FIT-only preflight of four compact learned world-model architectures.

No CAL, old TEST, ME064 revealed spray-nozzle outcomes, or new holdout read.
Original real BodenAI 5-task FIT source from registered ME064 identities.
No optimizer step is performed; gradients and operator-FLOPs are measured only.
This is NOT a trained comparison and no superiority claim follows.
"""
from __future__ import annotations
import argparse, json, platform
from pathlib import Path
import torch
from torch.profiler import ProfilerActivity, profile

from scripts.me064_mixed_real_fit import fit_metadata, batch
from scripts.me063_prefit_original_fit import model_for as original_model_for
from scripts.me063_prefit_original_fit import objective as physical_objective
from scripts.me060_train_trajectory import digest
from tac_osm.me065_micro_world_controls import MicroRegimeComparator

STUDY = "ME065_REAL_MULTITASK_COMPACT_WORLD_MODEL_COMPARATORS_PREFIT"
SEEDS=(65001,65002,65003,65004,65005)
# Configuration frozen before reading any source observations/outcomes.
CONFIG={
    "native_tac":(32,64),
    "gru_world":(44,82),
    "compact_rssm":(44,82),
    "selective_diagonal_ssm":(44,103),
}
TRAINING_UPDATES=600
ABS_MAX_RESOURCE_GAP=.01

def model_for(arm, identified):
    if arm in ("native_tac","gru_world"):
        return original_model_for(arm,identified)
    return MicroRegimeComparator(identified,arm,*CONFIG[arm])

def objective(model, data):
    if isinstance(model, MicroRegimeComparator):
        model.reset_auxiliary()
    loss=physical_objective(model,data)
    if isinstance(model, MicroRegimeComparator):
        loss=loss+model.auxiliary_objective()
    return loss

def measure(model, actual_batch):
    model.train()
    model.zero_grad(set_to_none=True)
    with profile(activities=[ProfilerActivity.CPU],with_flops=True) as profiler:
        loss=objective(model,actual_batch)
        loss.backward()
    return {
        "trainable_parameters":sum(p.numel() for p in model.parameters()),
        "supported_operator_forward_backward_flops_per_update":int(
            sum(event.flops for event in profiler.key_averages())),
        "initial_loss":float(loss.detach()),
        "all_trainable_gradients_finite_and_present":all(
            p.grad is not None and bool(torch.isfinite(p.grad).all())
            for p in model.parameters()),
        "no_optimizer_step":True,
    }

def main():
    cli=argparse.ArgumentParser()
    cli.add_argument("--source-cache",type=Path,required=True)
    cli.add_argument("--output",type=Path,required=True)
    args=cli.parse_args()
    torch.set_num_threads(1)
    registered, identified, source, receipt, orders=fit_metadata(args.source_cache)
    # A frozen original FIT source identity; no new physical data admitted.
    if receipt["fit_original_sha256"]!="4d01753b2de68ef2eb863b94aeb790f6b2f98c6068aede3bf60e315f72a8b776":
        raise RuntimeError("unrecognized original five-task FIT source")
    if registered["holdout_task"]!="install_spray_nozzle_onto_bottle":
        raise RuntimeError("historical ME064 registration changed")
    selections=orders["64001"][0]
    actual_batch=batch(source,selections)
    arms={}
    for arm in CONFIG:
        torch.manual_seed(SEEDS[0])
        model=model_for(arm,identified)
        arms[arm]=measure(model,actual_batch)
    sizes=[v["trainable_parameters"] for v in arms.values()]
    flops=[v["supported_operator_forward_backward_flops_per_update"] for v in arms.values()]
    param_gap=(max(sizes)-min(sizes))/max(sizes)
    flop_gap=(max(flops)-min(flops))/max(flops)
    result={
        "study":STUDY,"research_phase":"FROZEN_SOURCE_FIT_PREFLIGHT_ONLY",
        "training_started":False,"full_model_fits":0,"optimizer_updates":0,
        "independent_holdout_scored":False,
        "heldout_trajectory_files_opened":0,"cal_trajectory_files_opened":0,
        "already_revealed_me064_target_trajectory_files_opened":0,
        "original_real_fit_source_sha256":receipt["fit_original_sha256"],
        "fit_tasks":registered["training_tasks"],
        "fit_episode_count":len(source),
        "seeds_reserved":list(SEEDS),
        "matched_future_training_updates_per_seed":TRAINING_UPDATES,
        "profiled_real_factual_window_digest":digest(selections),
        "arms":arms,
        "parameter_relative_gap":param_gap,
        "supported_operator_flop_relative_gap":flop_gap,
        "all_active":all(v["all_trainable_gradients_finite_and_present"] for v in arms.values()),
        "numerical_parity_pass":bool(param_gap<ABS_MAX_RESOURCE_GAP and
                                      flop_gap<ABS_MAX_RESOURCE_GAP and
                                      all(v["all_trainable_gradients_finite_and_present"]
                                          for v in arms.values())),
        "profiler_limitation":"PyTorch-supported operator forward/backward only, not full optimizer/hardware FLOPs",
        "cpu_time_equality_required":False,
        "fair_comparison_qualified":False,
        "checkpoint_replay":"NOT_APPLICABLE_NO_TRAINING",
        "next_required_step":"separate prospective FIT-only fair comparison manifest and checkpoint byte storage audit, then equal original real FIT model training",
        "python":platform.python_version(),"pytorch":torch.__version__,
    }
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    print("ME065_AUTHENTIC_SOURCE_PREFLIGHT="+json.dumps({
        "param_gap":param_gap,"flop_gap":flop_gap,
        "per_arm":arms,"numeric_gate":result["numerical_parity_pass"],
        "models_fitted":0},sort_keys=True),flush=True)
    if not result["numerical_parity_pass"]:
        raise SystemExit("ME065_PRETRAIN_RESOURCE_OR_GRADIENT_PARITY_FAIL; NO MODEL FITS ALLOWED")
    print("ME065_PRETRAIN_RESOURCE_PARITY_PASS; NO MODEL FITS STARTED",flush=True)

if __name__=="__main__":
    main()
