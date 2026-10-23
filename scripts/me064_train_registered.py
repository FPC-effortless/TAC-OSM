"""ME064 real five-task FIT-only matched training. Never opens CAL or heldout.

Original model, optimizer and RNG checkpoint BYTES are committed in Git before
any attempt to measure transfer. A clean-checkout audit is compulsory.
"""
from __future__ import annotations
import json, os, platform, resource, time
from pathlib import Path
import numpy as np
import torch

from scripts.me064_mixed_real_fit import fit_metadata, batch, SEEDS
from scripts.me063_prefit_original_fit import CONFIG, model_for, objective, profile_once
from scripts.me057_train_fused_anchor_robot import sha_file, write_json
from scripts.me060_train_trajectory import digest, BATCH, HISTORY, HORIZON, UPDATES
from scripts.validate_research_parity import check_manifest

ROOT=Path("research/mathematical-engineering/me064")
OUT=Path("research/replay-bundles/ME064")


def train():
    torch.set_num_threads(1)
    if not (OUT/"UNTRAINED_STORAGE_PROBE.pt").exists() or not (OUT/"STORAGE_PROBE_MANIFEST.json").exists():
        raise RuntimeError("Git checkpoint original byte publication route not proven")
    manifest=json.loads((ROOT/"FAIR_COMPARISON.json").read_text())
    if manifest["study_type"]!="architecture_comparison":
        raise RuntimeError("study not correctly registered")
    if check_manifest(manifest,"preflight")["status"]!="PASS":
        raise RuntimeError("scientific comparison preflight rejected")
    measured=json.loads((ROOT/"PREFLIGHT_MEASURED.json").read_text())
    if measured["optimizer_updates"]!=0 or measured["cal_episodes_opened"]!=0 or measured["holdout_trajectories_opened"]!=0:
        raise RuntimeError("pretraining source isolation failed")
    if measured["supported_operator_flop_relative_gap"]>=.01 or measured["parameter_relative_gap"]>=.01:
        raise RuntimeError("per-arm source-fit resource mismatch")
    registered,identified,fit,original,orders=fit_metadata(Path(os.environ["ME064_PUBLISHER_CACHE"]))
    if original["fit_original_sha256"]!=measured["fit_original_sha256"] or (
        original["task_level_split_sha256"]!=measured["task_level_split_sha256"]):
        raise RuntimeError("original source FIT or metadata split changed")
    if registered["holdout_task"]!=manifest["prospective_untouched_target"]["task"]:
        raise RuntimeError("unseen transfer task changed after source registration")
    if any(original["per_seed_order_sha256"][str(s)]!=measured["per_seed_order_sha256"][str(s)] for s in SEEDS):
        raise RuntimeError("task balancing/source FIT sampling changed")
    ckpts=OUT/"checkpoints"
    ckpts.mkdir(parents=True,exist_ok=True)
    if any(ckpts.iterdir()):
        raise RuntimeError("refuse to overwrite previously published real model bytes")
    actual_records=[]
    for seed in SEEDS:
        picks=orders[str(seed)]
        for arm in CONFIG:
            receipt=next(r for r in manifest["arms"] if r["architecture"]==arm and r["seed"]==seed)
            if receipt["training_sample_order_sha256"]!=digest(picks):
                raise RuntimeError("compulsory training source order changed")
            torch.manual_seed(seed); np.random.seed(seed)
            model=model_for(arm,identified)
            count=sum(p.numel() for p in model.parameters())
            if count!=receipt["trainable_parameters"]:
                raise RuntimeError("matched parameter budget changed")
            optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
            traces=[]
            started=time.process_time()
            for step,pick in enumerate(picks,1):
                optimizer.zero_grad(set_to_none=True)
                loss=objective(model,batch(fit,pick))
                if not torch.isfinite(loss):
                    raise RuntimeError(f"nonfinite authentic multi-task FIT loss: {arm}/{seed}/{step}")
                loss.backward()
                norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
                if not torch.isfinite(norm):
                    raise RuntimeError(f"nonfinite authentic multi-task FIT gradient: {arm}/{seed}/{step}")
                optimizer.step()
                if step==1 or step%100==0:
                    traces.append({"step":step,"loss":float(loss.detach())})
                    print(json.dumps({"event":"ME064_ORIGINAL_FIT_MODEL","arm":arm,"seed":seed,
                                      **traces[-1]}),flush=True)
            cpu=time.process_time()-started
            actual_profile=profile_once(model,batch(fit,picks[-1]))
            measured_flops=actual_profile["profiled_forward_backward_flops"]*UPDATES
            file=ckpts/f"{arm}-seed{seed}.pt"
            torch.save({
                "experiment":manifest["study_id"],"training_complete":True,
                "architecture":arm,"seed":seed,
                "update_count":UPDATES,"training_source_sha256":original["fit_original_sha256"],
                "task_level_split_sha256":original["task_level_split_sha256"],
                "per_task_original_shas":original["per_task_fit_sha256"],
                "fit_order_sha256":digest(picks),
                "identified":{k:v.tolist() for k,v in identified.items()},
                "model":model.state_dict(),"optimizer":optimizer.state_dict(),
                "torch_rng_state":torch.get_rng_state(),"numpy_rng_state":np.random.get_state(),
                "model_config":list(CONFIG[arm]),"loss_trace":traces,
                "scoring_status":"NO_CAL_OR_HELDOUT_TRAJECTORY_OPENED",
            },file)
            measurements={
                "actual_steps":UPDATES,
                "actual_examples":UPDATES*BATCH,
                "actual_tokens":UPDATES*BATCH*(HISTORY+HORIZON),
                "training_flops":measured_flops,
                "optimizer_cpu_seconds":cpu,
                "peak_memory_bytes":int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024),
                "flops_profiler":"supported PyTorch operator original FIT full forward/backward; optimizer and missing ops excluded",
                "checkpoint_sha256":sha_file(file),
                "actual_sample_order_sha256":digest(picks),
                "evaluation_split_sha256":receipt["test_split_sha256"],
            }
            receipt["measured"]=measurements
            actual_records.append({
                "seed":seed,"arm":arm,"path":str(file),"bytes":file.stat().st_size,
                "checkpoint_sha256":measurements["checkpoint_sha256"],"measured":measurements
            })
            write_json(OUT/"CHECKPOINTS_FROZEN_PARTIAL.json",{
                "model_count":len(actual_records),"expected_count":10,
                "holdout_opened":False,"cal_opened":False,"original_checkpoints":actual_records})
            print(json.dumps({"event":"ME064_REAL_CHECKPOINT_NOT_YET_GIT",
                              "arm":arm,"seed":seed,"sha256":sha_file(file),"bytes":file.stat().st_size}),flush=True)
    final=check_manifest(manifest,"final")
    if final["status"]!="PASS": raise RuntimeError("final actual resource/compute parity failed; retain negative checkpoints")
    write_json(OUT/"FAIR_COMPARISON_FINAL.json",manifest)
    write_json(OUT/"TRAINING_RESULT.json",{
        "study":manifest["study_id"],"status":"FITTED_NOT_YET_INDEPENDENTLY_GIT_REPLAYED",
        "checkpoints":actual_records,"fit_source_sha256":original["fit_original_sha256"],
        "per_task_fit_sha256":original["per_task_fit_sha256"],
        "training_tasks":registered["training_tasks"],
        "unseen_target":registered["holdout_task"],
        "fit_original_episodes":len(fit),
        "source_calibration_episodes_opened":0,"holdout_original_trajectories_opened":0,
        "old_test_original_opened":False,"independent_test":False,
        "final_numerical_parity":final,"operator_and_optimizer_all_flops_certified":False,
        "replay_status":"AWAITING_ORIGINAL_GIT_CLEAN_CHECKOUT",
        "python":platform.python_version(),"torch":torch.__version__,
    })
    print("ME064_TEN_REAL_FIT_MODELS_COMPLETE_CAL_AND_UNSEEN_TARGET_UNOPENED",flush=True)


if __name__=="__main__":
    train()
