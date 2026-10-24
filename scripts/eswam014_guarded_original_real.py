"""ESWAM014 guarded real native intelligence study; no pseudo-action training.

Reuses exactly the certified original INTERACTION source-only reader and real
256-step joint-actor input contract of ESWAM012. This is a NEW registered,
single-native-arm source study, not retroactive certification of ESWAM012.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import random
import time

import torch
from scripts.eswam012_train_original_multiagent import (
    original_cases, stack, physical_loss, all_original_case_rows, score_one,
)
from scripts.validate_research_parity import check_manifest
from tac_osm.parameter_budget import (
    budgeted_optimizer, checked_optimizer_step, save_training_checkpoint,
    file_sha, validate_runtime_evidence,
)
from tac_osm.variants.eswam014_quadratic_motion import QuadraticMotionNative

ROOT = Path("research/mathematical-engineering/eswam014")
MANIFEST = ROOT / "FAIR_COMPARISON.json"
OUT = Path("research/replay-bundles/ESWAM014")
SEEDS = (87001, 87002)
UPDATES = 320
BATCH = 8
PARAMETERS = 101057
ARCH = "eswam014_quadratic_kinetic_relevance_native"


def preflight():
    model = QuadraticMotionNative()
    active = sum(p.numel() for p in model.parameters() if p.requires_grad)
    deployed = sum(p.numel() for p in model.parameters())
    if (active, deployed) != (PARAMETERS, PARAMETERS):
        raise RuntimeError(f"live deployed/active parameter mismatch {active}/{deployed}")
    manifest = json.loads(MANIFEST.read_text())
    verdict = check_manifest(manifest, "preflight")
    if verdict["status"] != "PASS_PREFIT_SINGLE_MODEL":
        raise RuntimeError("source-native compulsory signed preflight failed")
    registered = {(r["seed"], r["architecture"], r["trainable_parameters"],
                   r["deployed_parameters"]) for r in manifest["models"]}
    expected = {(s, ARCH, PARAMETERS, PARAMETERS) for s in SEEDS}
    if registered != expected:
        raise RuntimeError("post-registration source model/seed/parameter tampering")
    return active


def fit_original_real(archive: Path, out: Path):
    preflight()
    data, excluded = original_cases(archive)
    fit = sum((data[rec] for rec in (0, 1, 2)), [])
    dev = data[3]
    original_fit_source_schedule = hashlib.sha256(json.dumps(
        [(r["recording"], r["frame"], r["actor_ids"]) for r in fit],
        sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    out.mkdir(parents=True, exist_ok=True)
    original_receipts = []
    torch.set_num_threads(2)
    for seed in SEEDS:
        rng = random.Random(seed)
        # Source anchors and schedules are entirely FIT-derived, fixed before
        # opening any prospective actual future labels in exposed DEV.
        schedule = [[rng.randrange(len(fit)) for _ in range(BATCH)]
                    for _ in range(UPDATES)]
        schedule_sha = hashlib.sha256(json.dumps(
            schedule, separators=(",", ":")).encode()).hexdigest()
        torch.manual_seed(seed)
        model = QuadraticMotionNative()
        receipt_file = out/f"live-guard-{seed}.json"
        checkpoint = out/f"seed-{seed}.pt"
        if receipt_file.exists() or checkpoint.exists():
            raise RuntimeError("STOP: original guarded checkpoint or receipt already exists")
        optimizer = budgeted_optimizer(
            model, torch.optim.AdamW, model.parameters(),
            lr=.001, weight_decay=.00001,
            budget_manifest=MANIFEST, budget_seed=seed,
            budget_architecture=ARCH, budget_receipt=receipt_file)
        model.train()
        losses = []
        start = time.process_time()
        for update, indices in enumerate(schedule, 1):
            x, prior, target, active = stack([fit[k] for k in indices])
            optimizer.zero_grad(set_to_none=True)
            prediction, scale = model(x, prior, active)
            loss = physical_loss(prediction, scale, target, active)
            if not bool(torch.isfinite(loss)):
                raise RuntimeError(f"source factual loss nonfinite at {update}")
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if not bool(torch.isfinite(norm)):
                raise RuntimeError("nonfinite full model observed gradient norm")
            checked_optimizer_step(optimizer)
            losses.append(float(loss.detach()))
            if update in (1, 80, 160, 240, 320):
                print("ESWAM014_REAL_FIT_TRACE="+json.dumps(
                    {"seed":seed,"updates":update,"observed_train_loss":losses[-1]},
                    sort_keys=True), flush=True)
        process_seconds = time.process_time() - start
        frozen = {
            "seed":seed,"architecture":ARCH,
            "original_source_archive_sha256":
                "9aa78102006dc8d8a64675cf64eba5264085cfb1c006b1f4d0e856d94e657aec",
            "original_fit_samples_sha256":original_fit_source_schedule,
            "training_schedule_sha256":schedule_sha,
            "actual_guarded_optimizer_steps":UPDATES,
            "weights":model.state_dict(),
            "optimizer":optimizer.state_dict(),
            "torch_rng_after_fit":torch.get_rng_state(),
            "python_rng_after_fit":rng.getstate(),
            "training_loss_first":losses[0],
            "training_loss_final":losses[-1],
        }
        save_training_checkpoint(frozen, checkpoint)
        receipt = json.loads(receipt_file.read_text())
        if receipt["status"] != "CHECKPOINT_VERIFIED":
            raise RuntimeError("STOP: entire whole-model checkpoint live budget verification failed")
        cksha = file_sha(checkpoint)
        evidence = {
            "receipt_path":str(receipt_file),
            "receipt_sha256":file_sha(receipt_file),
            "checkpoint_path":str(checkpoint),
            "checkpoint_sha256":cksha,
        }
        # Check the independent serialized real model and optimizer receipt
        # via the repository's established compulsory guard, not model claims.
        guard_id = validate_runtime_evidence({
            "seed":seed,"architecture":ARCH,
            "trainable_parameters":PARAMETERS,
            "deployed_parameters":PARAMETERS,
            "checkpoint_sha256":cksha,
            "_prefit_registration_sha256":file_sha(MANIFEST),
            "_expected_optimizer_steps":UPDATES,
            "runtime_evidence":evidence,
        })
        model.eval()
        dev_rows = all_original_case_rows(model,dev,"eswam014")
        original_cases_file = out/f"seed-{seed}-ORIGINAL-REAL-DEV-CASES.json"
        original_cases_file.write_text(json.dumps(dev_rows,indent=2)+"\n")
        summaries = [
            score_one(dev_rows,seed,ARCH,h,schedule_sha,cksha,
                      process_seconds,losses[0],losses[-1])
            for h in (10,30)]
        for record in summaries:
            record["guard_verification_run_id"]=guard_id
            print("ESWAM014_REAL_SOURCE_SCORE="+
                  json.dumps(record,sort_keys=True),flush=True)
        original_receipts.append({
            "seed":seed,"architecture":ARCH,
            "fitted_checkpoint_sha256":cksha,
            "original_real_DEV_predictions_sha256":file_sha(original_cases_file),
            "live_budget_receipt_sha256":file_sha(receipt_file),
            "actual_guarded_steps":receipt["successful_optimizer_steps"],
            "compulsory_guard_verified":True,
            "training_schedule_sha256":schedule_sha,
            "replayed_live_guard_id":guard_id,
            "fit_CPU_seconds":process_seconds,
            "scores":summaries,
        })
    result = {
        "study":"ESWAM014_GUARDED_MOTION_RELEVANCE_NATIVE_REAL_TRAFFIC",
        "source_archive_sha256":"9aa78102006dc8d8a64675cf64eba5264085cfb1c006b1f4d0e856d94e657aec",
        "real_training_origins":len(fit),"real_exposed_DEV_origins":len(dev),
        "complete_original_observed_history":256,
        "source_fit_member_indices":[0,1,2],
        "source_EXPOSED_DEV_member_index":3,
        "source_recording_same_physical_site":True,
        "source_independent_site_TEST_exists":False,
        "trials_or_model_selection_on_DEV":0,
        "pre_registered_active_and_deployed_params":PARAMETERS,
        "loss_and_training_optimizer_updates":UPDATES,
        "training_FLOPs_equal_rival_certified":False,
        "architecture_superiority_allowed":False,
        "original_real_source_only":True,"fitted_on_synthetic":False,
        "previous_ESWAM012_checkpoint_guards_failed_retroactively_fixed":False,
        "original_unaligned_actor_anchors_rejected":excluded,
        "original_fit_sample_digest":original_fit_source_schedule,
        "original_guarded_fits":original_receipts,
    }
    (out/"FINAL_RESULTS.json").write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    print("ESWAM014_FINAL_PASS="+json.dumps({
        "real_fitted_checkpoints":len(original_receipts),
        "both_full_model_optimizer_runtime_guards_passed":True,
        "original_trained_source_samples":len(fit)},sort_keys=True),flush=True)


if __name__ == "__main__":
    p=argparse.ArgumentParser()
    p.add_argument("mode",choices=["preflight","fit"])
    p.add_argument("--archive",type=Path)
    p.add_argument("--out",type=Path,default=OUT)
    args=p.parse_args()
    if args.mode=="preflight":
        print("ESWAM014_GUARDED_MODEL_PREFLIGHT="+str(preflight()),flush=True)
    else:
        if args.archive is None:
            raise ValueError("original publisher real source required")
        fit_original_real(args.archive,args.out)
