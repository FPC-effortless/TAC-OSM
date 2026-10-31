"""LANGCAP104: fail-closed real-Cpython byte training worker.

No model fits can start from LANGCAP103 REVIEW_DRAFT registrations.
Requires a separately committed frozen FAIR_COMPARISON registration,
operator-pinned external Ed25519 attestation, explicit protected-GPU
enablement, and verified FIT-only corpus materialization. Never reads DEV/TEST.

A trained artifact is not an independently qualified model until a separate
postfit independent outcome/reward/access attestation passes the canonical gate.
"""
from __future__ import annotations
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import random
import sys
import tarfile
import time

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/"src"))
sys.path.insert(0,str(ROOT/"research/langcap103"))
EXPECTED_SOURCE_SHA="94fc488af0e975c0643e1d566d0468a4091199fdae02b159c8669a01bebef842"
ELIGIBLE_MODELS={"tac_associative_stack","causal_transformer","stacked_gru"}
SEEDS={10301,10302,10303}

class NotAuthorized(ValueError):
    pass

def require(ok,message):
    if not ok:raise NotAuthorized(message)

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def load_json(path):
    return json.loads(Path(path).read_text())

def verify_authorization(manifest_path, evidence_path=None):
    """Gate first; no data or Torch import, optimizer, or network before this."""
    manifest_path=Path(manifest_path)
    require(manifest_path.name=="FAIR_COMPARISON.json" and
            "langcap104" in manifest_path.parts,
            "No training from registration drafts or alternate manifest paths")
    require(manifest_path.is_file(),"Frozen FAIR_COMPARISON registration missing")
    manifest=load_json(manifest_path)
    require(manifest.get("study_type")=="single_model_training" and
            manifest.get("training_enabled") is True and
            manifest.get("registration_status")=="FROZEN_APPROVED_AUTHENTIC_FIT",
            "Registration is not the approved frozen trainable contract")
    require(manifest.get("real_world_source",{}).get("sha256")==EXPECTED_SOURCE_SHA,
            "Authentic CPython source identity differs")
    require(manifest.get("no_architecture_superiority_claim") is True,
            "Independent fits do not authorize architecture superiority")
    compute=manifest.get("compute",{})
    require(type(compute.get("approved_gpu_minutes")) is int and
            compute["approved_gpu_minutes"]>0 and
            type(compute.get("training_steps")) is int and
            compute["training_steps"] in (2560,12800,25600) and
            isinstance(compute.get("t4_throughput_verified"),dict) and
            isinstance(compute.get("full_training_flops_verified"),dict),
            "Approved GPU quota, frozen steps and measured compute receipts required")
    require(os.getenv("LANGCAP104_GPU_EXECUTION_ENABLED")=="true",
            "Operator has not enabled protected GPU execution")
    required_sha=os.getenv("LANGCAP104_FROZEN_MANIFEST_SHA256","")
    require(len(required_sha)==64 and sha(manifest_path)==required_sha,
            "Operator-pinned frozen manifest hash missing or changed")
    key=os.getenv("TAC_OSM_REWARD_VERIFIER_PUBLIC_KEY_HEX")
    require(isinstance(key,str) and len(key)==64,
            "Independently provisioned verification key missing")
    require(evidence_path is not None and Path(evidence_path).is_file(),
            "Independent signed prefit attestation missing")
    from scripts.validate_research_parity import check_manifest
    result=check_manifest(manifest,"preflight",reward_attestation=load_json(evidence_path),
                          trusted_reward_public_key_hex=key)
    require(result["reward_integrity"]["status"]=="PASS_SIGNED_EXTERNAL_AUDIT",
            "External source/outcome/ACL audit did not pass")
    return manifest,required_sha

def fit_only(receipt_path,archive_path):
    """Read FIT-only staged archive; DEV bytes must not be delivered here."""
    from source import canonical,digest,git_blob
    receipt=load_json(receipt_path)
    require(receipt["source_sha256"]==EXPECTED_SOURCE_SHA and
            digest(canonical(receipt["identity"]))==EXPECTED_SOURCE_SHA,
            "Publisher source identity mismatch")
    originals={r["path"]:r for r in receipt["identity"]["files"] if r["split"]=="FIT"}
    require(len(originals)==560,"Expected exact 560 authenticated FIT files")
    with tarfile.open(archive_path,"r") as tar:
        actual={}
        for member in tar:
            require(member.isfile() and member.name in originals and
                    member.name not in actual and not member.issym() and
                    not member.islnk(),"Unexpected or duplicated file in FIT-only archive")
            raw=tar.extractfile(member).read()
            row=originals[member.name]
            require(len(raw)==row["bytes"] and digest(raw)==row["sha256"] and
                    git_blob(raw)==row["git_blob"],"FIT original byte hash mismatch")
            actual[member.name]=raw
    require(set(actual)==set(originals),"FIT-only corpus missing original files")
    return receipt,actual

def model_instance(config,target):
    import torch
    import preflight
    native=preflight.load_native()
    return preflight.build(config,target,native,preflight.comparator_classes(torch))

def train(args):
    manifest,manifest_sha=verify_authorization(args.manifest,args.prefit_evidence)
    require(args.output is not None and not args.output.exists(),
            "Training destination must be absent; never overwrite weights")
    require(args.arm in ELIGIBLE_MODELS and args.seed in SEEDS,
            "Unexpected architecture/seed outside registered matrix")
    entries=[m for m in manifest["models"]
             if (m["variant"],m["seed"])==(args.arm,args.seed)]
    require(len(entries)==1 and len(manifest["models"])==9,
            "All nine planned fits must be preregistered; one selected fit per job")
    allowed={(x["variant"],x["seed"]) for x in manifest["models"]}
    require(allowed=={(arm,seed) for arm in ELIGIBLE_MODELS for seed in SEEDS},
            "Nine-run fixed matrix incomplete")
    row=entries[0]
    require(args.device=="cuda","Actual training requires authorized CUDA")
    # Original FIT-only data are available only AFTER signed gate above.
    source,data=fit_only(args.receipt,args.fit_archive)
    require(source["source_sha256"]==manifest["real_world_source"]["sha256"],
            "Signed source binding does not match FIT-only bytes")
    from prepare_registration import sample_stream,source_blocks,SEQUENCE,BATCH
    require(SEQUENCE==256 and BATCH==32,"Frozen sample/exposure geometry differs")
    blocks=source_blocks(source)
    require(blocks and all(x[0] in data for x in blocks),
            "FIT-only stream references foreign or DEV data")
    import torch
    import torch.nn.functional as F
    require(torch.cuda.is_available() and torch.cuda.device_count()>=2 and
            all("T4" in torch.cuda.get_device_name(i) for i in range(2)),
            "Exactly authorized dual-T4 execution venue required")
    torch.manual_seed(args.seed);random.seed(args.seed)
    device=torch.device("cuda:0")
    target=manifest["parameter_scale"]["target_trainable_parameters"]
    model=model_instance(row["configuration"],target).float().to(device).train()
    actual_train=sum(p.numel() for p in model.parameters() if p.requires_grad)
    actual_total=sum(p.numel() for p in model.parameters())
    require((actual_train,actual_total)==
            (row["trainable_parameters"],row["deployed_parameters"]),
            "Actual model is not the preregistered exact parameter instance")
    optimizer_policy=manifest["optimizer_proposal"]
    require(optimizer_policy["name"]=="AdamW" and
            optimizer_policy.get("status")=="REVIEWED_AND_FROZEN" and
            optimizer_policy.get("precision")=="float32",
            "Optimizer and precision must be independently frozen")
    optimizer=torch.optim.AdamW(model.parameters(),lr=optimizer_policy["base_lr"],
       betas=tuple(optimizer_policy["betas"]),eps=optimizer_policy["eps"],
       weight_decay=optimizer_policy["weight_decay"])
    stream=sample_stream(blocks,args.seed)
    total=manifest["compute"]["training_steps"]
    maximum=manifest["compute"]["approved_gpu_minutes"]*60
    started=time.monotonic();losses=[];steps=0;observed=0
    args.output.mkdir(parents=True)
    def snapshot(status):
        payload={"model":model.state_dict(),"optimizer":optimizer.state_dict(),
                 "rng_cpu":torch.get_rng_state(),"rng_cuda":torch.cuda.get_rng_state_all(),
                 "manifest_sha256":manifest_sha,"source_sha256":EXPECTED_SOURCE_SHA,
                 "arm":args.arm,"seed":args.seed,"steps":steps,"status":status}
        path=args.output/("TRAINED.pt" if status=="TRAINING_COMPLETED_NOT_QUALIFIED" else "PARTIAL.pt")
        with path.open("xb") as out:torch.save(payload,out)
        return sha(path)
    try:
        for index in range(1,total+1):
            require(time.monotonic()-started < maximum,
                    "Approved GPU budget expired; retain partial weights only")
            batch_rows=[next(stream) for _ in range(BATCH)]
            tokens=[];targets=[]
            for name,offset in batch_rows:
                raw=data[name][offset:offset+SEQUENCE+1]
                require(len(raw)==SEQUENCE+1,"Corpus chunk exceeds file boundary")
                tokens.append(list(raw[:-1]));targets.append(list(raw[1:]))
            x=torch.tensor(tokens,dtype=torch.long,device=device)
            y=torch.tensor(targets,dtype=torch.long,device=device)
            optimizer.zero_grad(set_to_none=True)
            output=model(x,owners=tuple(f"fit-{i}" for i in range(BATCH))) if args.arm=="tac_associative_stack" else model(x)
            logits=output[0] if isinstance(output,tuple) else output
            require(logits.shape==(BATCH,SEQUENCE,256),"Actual causal byte logits shape mismatch")
            loss=F.cross_entropy(logits.reshape(-1,256),y.reshape(-1))
            require(bool(torch.isfinite(loss)),"Nonfinite FIT objective")
            loss.backward()
            require(all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in model.parameters() if p.requires_grad),
                    "Missing or nonfinite FIT parameter gradient")
            optimizer.step()
            steps=index;observed+=BATCH*SEQUENCE
            losses.append(float(loss.detach().cpu()))
            if index in (1,256,2560,12800,25600):
                print(json.dumps({"status":"FIT_PROGRESS_NOT_QUALIFIED","arm":args.arm,
                    "seed":args.seed,"steps":steps,"loss":losses[-1],"observed_bytes":observed}),flush=True)
        ck=snapshot("TRAINING_COMPLETED_NOT_QUALIFIED")
        status="TRAINING_COMPLETED_NOT_QUALIFIED"
    except Exception as exc:
        status="FAILED_OR_INCOMPLETE_NO_DEV_PERMITTED"
        ck=snapshot(status) if steps else None
        (args.output/"FAILURE.json").write_text(json.dumps(
            {"status":status,"error_type":type(exc).__name__,
             "error":str(exc),"steps_completed":steps},indent=2)+"\n")
    report={"status":status,"arm":args.arm,"seed":args.seed,"steps":steps,
            "required_steps":total,"losses":losses,
            "source_sha256":EXPECTED_SOURCE_SHA,"manifest_sha256":manifest_sha,
            "checkpoint_sha256":ck,"observed_fit_bytes":observed,
            "training_elapsed_seconds":time.monotonic()-started,
            "no_dev_scoring":True,"protected_test_access":False,
            "postfit_independent_attestation":"NOT_PERFORMED",
            "architecture_superiority":False}
    (args.output/"FIT_RECEIPT.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps({"status":status,"steps":steps,"checkpoint_sha256":ck}),flush=True)
    return 0 if status=="TRAINING_COMPLETED_NOT_QUALIFIED" else 2

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--manifest",type=Path,required=True)
    p.add_argument("--prefit-evidence",type=Path)
    p.add_argument("--receipt",type=Path)
    p.add_argument("--fit-archive",type=Path)
    p.add_argument("--arm",choices=sorted(ELIGIBLE_MODELS))
    p.add_argument("--seed",type=int)
    p.add_argument("--output",type=Path)
    p.add_argument("--device",choices=("cuda",),default="cuda")
    p.add_argument("--check-authorization-only",action="store_true")
    args=p.parse_args()
    try:
        if args.check_authorization_only:
            verify_authorization(args.manifest,args.prefit_evidence)
            print("PREFIT_AUTHORIZATION_VERIFIED_NO_OPTIMIZER")
            return 0
        require(args.receipt and args.fit_archive and args.arm and args.seed and args.output,
                "Incomplete training input fields")
        return train(args)
    except (NotAuthorized,ValueError,KeyError,OSError) as exc:
        print(json.dumps({"status":"TRAINING_BLOCKED_NO_OPTIMIZER","reason":str(exc)}),
              file=sys.stderr)
        return 2

if __name__=="__main__":
    raise SystemExit(main())
