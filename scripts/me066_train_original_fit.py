"""ME066: four-arm paired original five-task real robot FIT training.

Never loads source CAL or any heldout robot trajectory. First training is
forbidden without a Git-committed, numeric validated FAIR_COMPARISON, frozen
metadata task selection and repository-storage proof. Preserve every fitted
model+optimizer+RNG byte, even if final resource/accuracy gates fail.
"""
from __future__ import annotations
import hashlib,json,os,platform,resource,time
from pathlib import Path
import numpy as np
import torch

from scripts.me064_mixed_real_fit import prepare_fit,schedules,batch
from scripts.me065_prefit_original_fit import CONFIG,SEEDS,model_for,objective,measure
from scripts.me060_train_trajectory import digest
from scripts.validate_research_parity import check_manifest

ROOT=Path("research/mathematical-engineering/me066")
OUT=Path("research/replay-bundles/ME066")
STUDY="ME066_FOUR_ARM_GENUINE_ORIGINAL_ROBOT_FIT_EQUAL_BUDGET"

def digest_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def write_json(path, value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,sort_keys=True)+"\n")

def train():
    torch.set_num_threads(1)
    manifest=json.loads((ROOT/"FAIR_COMPARISON.json").read_text())
    registration=json.loads((ROOT/"TASK_LEVEL_SPLIT.json").read_text())
    permission=json.loads((ROOT/"TRAINING_PERMISSION.json").read_text())
    probe=json.loads((OUT/"STORAGE_PROBE_MANIFEST.json").read_text())
    if manifest["study_id"]!=STUDY or not manifest.get("training_authorized"):
        raise RuntimeError("ME066 training not authorized by the frozen real source manifest")
    if not permission.get("train_original_FIT_only") or permission.get("source_manifest_sha256")!=digest_file(ROOT/"FAIR_COMPARISON.json"):
        raise RuntimeError("original preregistration digest not bound to training permission")
    if (permission["target_trajectory_access_allowed"] or
            registration["heldout_original_trajectory_files_opened"]!=0 or
            registration["source_CAL_trajectory_files_opened"]!=0):
        raise RuntimeError("no target or CAL reads authorized before every Git checkpoint freezes")
    if probe.get("status")!="GIT_STORAGE_PROBE_PASS" or not (OUT/"UNTRAINED_STORAGE_PROBE.pt").exists():
        raise RuntimeError("Git-owned original checkpoint byte storage has not been preproven")
    if digest_file(OUT/"UNTRAINED_STORAGE_PROBE.pt")!=probe["sha256"]:
        raise RuntimeError("checkpoint storage probe binary bytes changed")
    if check_manifest(manifest,"preflight")["status"]!="PASS":
        raise RuntimeError("parameter/real source/operator FLOP parity registration invalid")
    pre=json.loads(Path("research/mathematical-engineering/me065/PREFLIGHT_MEASURED.json").read_text())
    if not pre["numerical_parity_pass"] or not pre["all_active"] or pre["full_model_fits"]!=0:
        raise RuntimeError("original real FIT prefit failed or changed")
    plan,identified,fit,source_hash,per_task=prepare_fit(Path(os.environ["ME066_FIT_CACHE"]))
    if source_hash!=registration["source_FIT_hash"] or source_hash!=manifest["real_world_source"]["sha256"]:
        raise RuntimeError("source FIT authentic hash changed")
    if set(registration["training_task_names"])!=set(plan["training_tasks"]):
        raise RuntimeError("source tasks changed")
    if registration["heldout_task_name"] in plan["training_tasks"] or registration["heldout_task_name"]=="install_spray_nozzle_onto_bottle":
        raise RuntimeError("new test task is not held out from earlier revealed test")
    if registration["training_source_task_sha256"]!=per_task:
        raise RuntimeError("source original FIT per-task membership changed")
    ckpts=OUT/"checkpoints"
    ckpts.mkdir(parents=True,exist_ok=True)
    if any(ckpts.iterdir()):
        raise RuntimeError("original ME066 models already fit: refuse retraining and changing provenance")
    report=[]
    try:
        for seed in SEEDS:
            picks=schedules(fit,seed,plan["training_tasks"])
            if registration["training_seed_sample_digests"][str(seed)]!=digest(picks):
                raise RuntimeError("paired seed FIT windows do not match frozen prospective schedule")
            for arm in CONFIG:
                receipt=next(r for r in manifest["arms"] if r["architecture"]==arm and r["seed"]==seed)
                if receipt["training_sample_order_sha256"]!=digest(picks):
                    raise RuntimeError("source original per-arm FIT sampling changed")
                torch.manual_seed(seed)
                np.random.seed(seed)
                model=model_for(arm,identified)
                if sum(p.numel() for p in model.parameters())!=receipt["trainable_parameters"]:
                    raise RuntimeError("candidate active parameter count changed since prefit")
                optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
                started=time.process_time()
                trace=[]
                for step,pick in enumerate(picks,1):
                    optimizer.zero_grad(set_to_none=True)
                    loss=objective(model,batch(fit,pick))
                    if not bool(torch.isfinite(loss)):
                        raise RuntimeError(f"nonfinite genuine FIT loss for {arm} seed {seed} update {step}")
                    loss.backward()
                    norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
                    if not bool(torch.isfinite(norm)):
                        raise RuntimeError("nonfinite authentic FIT gradient")
                    optimizer.step()
                    if step==1 or step%100==0:
                        record={"step":step,"source_fit_loss":float(loss.detach())}
                        trace.append(record)
                        print(json.dumps({"event":"ME066_ORIGINAL_REAL_FIT","arm":arm,"seed":seed,**record}),flush=True)
                cpu=time.process_time()-started
                actual=measure(model,batch(fit,picks[-1]))
                measured_flops=int(actual["supported_operator_forward_backward_flops_per_update"])*600
                file=ckpts/f"{arm}-seed{seed}.pt"
                torch.save({
                    "study":STUDY,"training_complete":True,
                    "arm":arm,"seed":seed,"training_updates":600,
                    "model_config":list(CONFIG[arm]),"model":model.state_dict(),
                    "optimizer":optimizer.state_dict(),
                    "torch_rng_state":torch.get_rng_state(),
                    "numpy_rng_state":np.random.get_state(),
                    "FIT_source_sha256":source_hash,
                    "source_task_shas":per_task,
                    "paired_order_sha256":digest(picks),
                    "heldout_task_metadata_hash":registration["heldout_member_digest"],
                    "heldout_trajectory_files_opened":0,
                    "source_CAL_trajectory_files_opened":0,
                    "identified":{k:v.tolist() for k,v in identified.items()},
                    "loss_trace":trace,
                    "scoring_status":"NO_SOURCE_CAL_OR_FUTURE_TARGET_OPENED",
                    },file)
                measurement={
                    "actual_steps":600,"actual_examples":4800,"actual_tokens":384000,
                    "training_flops":measured_flops,
                    "optimizer_cpu_seconds":cpu,
                    "peak_memory_bytes":int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024),
                    "flops_profiler":"PyTorch with_flops supported operator forward/backward on original real source FIT actual final model; optimizer/missing ops excluded",
                    "checkpoint_sha256":digest_file(file),
                    "actual_sample_order_sha256":digest(picks),
                    "evaluation_split_sha256":receipt["test_split_sha256"],
                }
                receipt["measured"]=measurement
                row={
                    "arm":arm,"seed":seed,"path":str(file),"bytes":file.stat().st_size,
                    "checkpoint_sha256":measurement["checkpoint_sha256"],"measured":measurement,
                }
                report.append(row)
                write_json(OUT/"CHECKPOINTS_FROZEN_PARTIAL.json",{
                    "completed_count":len(report),"expected_count":20,
                    "all_targets_unopened":True,
                    "frozen_original_models":report,
                    "source_sha256":source_hash})
                print(json.dumps({"event":"ME066_CHECKPOINT_ORIGINAL_BYTES_NOT_YET_GIT","arm":arm,
                                  "seed":seed,"sha256":measurement["checkpoint_sha256"]}),flush=True)
        final=check_manifest(manifest,"final")
        if final["status"]!="PASS":
            raise RuntimeError("actual postfit scientific compute or data parity failed")
        write_json(OUT/"FAIR_COMPARISON_FINAL.json",manifest)
        write_json(OUT/"TRAINING_RESULT.json",{
            "study":STUDY,"status":"ORIGINAL_FIT_MODELS_NOT_YET_CLEAN_GIT_REPLAYED",
            "source_hash":source_hash,"target_task_name":registration["heldout_task_name"],
            "original_checkpoints":report,
            "final_numeric_parity":final,
            "target_trajectory_files_opened":0,
            "source_cal_trajectory_files_opened":0,
            "operator_and_optimizer_all_flops_certified":False,
            "model_git_replay":"PENDING_SEPARATE_GIT_CHECKOUT",
            "real_task_transfer_result":"NOT_RUN",
            "python":platform.python_version(),"torch":torch.__version__,
        })
        print("ME066_ORIGINAL_20_FITTED_CHECKPOINTS_READY_FOR_GIT_STORAGE_NO_TEST_OPENED",flush=True)
    except Exception:
        write_json(OUT/"TRAINING_FAILED_PARTIAL.json",{
            "study":STUDY,"status":"FAILED_NOT_SCORED",
            "original_checkpoint_count":len(report),"original_checkpoints":report,
            "heldout_trajectories_opened":0,"source_cal_trajectories_opened":0,
            "fit_partial_original_bytes_must_remain_recoverable":True})
        raise

if __name__=="__main__":
    train()
