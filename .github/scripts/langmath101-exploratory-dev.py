"""LANGMATH101 mixed-provenance exploratory DEV study.

This tool NEVER changes or relabels original run 41174892770 and NEVER
produces the original SEALED_ALL_FITS.json. It creates an explicitly posthoc
cohort using exactly 135 original successful fit archives + 3 independently
identified amended-time recovery archives. All 138 checkpoints are inspected
before any DEV model evaluation. The original evaluator/aggregator source is
kept byte-for-byte frozen; a bounded, audited evaluation-gate substitution is
used only in this new branch's explicit exploratory workflow.
"""
import argparse
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[2]
STUDY = ROOT / "research/mathematical-engineering/langmath101"
ORIGINAL_RUN = 41174892770
RECOVERY_RUN = 41179897888
FAILURES = {
    ("a5_adaptive", 10102, "counted_arithmetic_matched"),
    ("a5_adaptive", 10103, "counted_arithmetic_matched"),
    ("a5_adaptive", 10103, "parameter_exposure_matched"),
}
STATUS = "SEALED_EXPLORATORY_MIXED_PROVENANCE_BEFORE_DEV"

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as f:
        json.dump(value, f, indent=2, sort_keys=True, allow_nan=False)
        f.write("\n")

def read(path):
    return json.loads(Path(path).read_text())

def mod(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = m
    spec.loader.exec_module(m)
    return m

def modules():
    return mod("langmath101_exploratory_common", STUDY/"common.py"), mod(
        "langmath101_exploratory_finalizer", STUDY/"finalize.py")

def amended_script():
    return mod("langmath101_exploratory_amendment", ROOT/".github/scripts/langmath101-time-amendment.py")

def identities(manifest):
    rows = {(r["architecture"],r["seed"],r["lane"]):r for r in manifest["arms"]}
    if len(rows) != 138 or not FAILURES <= set(rows):
        raise ValueError("Expected all 138 frozen identities")
    return rows

def archive_name(prefix, key):
    return prefix + "-" + str(key[0]) + "-" + str(key[1]) + "-" + key[2]

def verify_recovery(directory, key, expected_worker, expected_effective):
    note = read(directory/"POSTHOC_AMENDMENT.json")
    expected = {
        "original_run_id": ORIGINAL_RUN,
        "original_registered_limit_seconds": 2700,
        "posthoc_limit_seconds": 4800,
        "original_worker_sha256": expected_worker,
        "effective_worker_sha256": expected_effective,
        "original_fit_eligibility": False,
        "protected_TEST_access": False,
        "DEV_access": False,
        "outcome": "TRAINING_COMPLETED_EXPLORATORY_ONLY",
        "architecture": key[0],
        "seed": key[1],
        "lane": key[2],
    }
    for k,v in expected.items():
        if note.get(k) != v:
            raise ValueError("Posthoc provenance differs for %s: %s"%(key,k))
    if not (directory/"POSTHOC_MODEL.pt").is_file():
        raise ValueError("Missing exploratory recovered checkpoint")
    if sha(directory/"POSTHOC_MODEL.pt") != note.get("original_checkpoint_sha256"):
        raise ValueError("Recovery checkpoint bytes do not match amendment")
    if (directory/"MODEL.pt").exists() or (directory/"FIT_RECEIPT.json").exists():
        raise ValueError("Recovery archive must not masquerade as original")
    return note

def seal(a):
    if a.output.exists():
        raise FileExistsError("Do not overwrite previous exploratory outputs")
    C, F = modules()
    manifest = read(STUDY/"FAIR_COMPARISON.json")
    rows = identities(manifest)
    gate = C.local("owner_two_lane")
    gate.validate_program(manifest,"preflight",root=C.ROOT)
    profile = read(C.ROOT/manifest["operator_profile_path"])
    if profile["source_hashes"] != C.source_hashes():
        raise ValueError("Frozen source mismatch before exploratory DEV")
    recovery = amended_script()
    original_worker, effective_worker, _, _ = recovery.effective_source()
    if sha(STUDY/"train.py") != hashlib.sha256(original_worker).hexdigest():
        raise ValueError("Original worker changed")
    a.output.mkdir(parents=True)
    selected, verified, full_rows = [], [], copy.deepcopy(manifest)
    seen_original, seen_recovery = set(), set()
    for row in full_rows["arms"]:
        key=(row["architecture"],row["seed"],row["lane"])
        original=a.original/archive_name("langmath101-fit",key)
        if not original.is_dir():
            raise ValueError("Missing original original-run artifact %s"%(key,))
        seen_original.add(original.name)
        failure=key in FAILURES
        if failure:
            if not (original/"FAILURE.json").is_file() or not (original/"FAILED_MODEL.pt").is_file():
                raise ValueError("Missing original failed-state evidence %s"%(key,))
            report=read(original/"FAILURE.json")
            if report.get("status")!="FAILED_RETAINED_NOT_EVALUABLE" or report.get("exception")!="TimeoutError":
                raise ValueError("Incorrect original failed-state evidence %s"%(key,))
            if (original/"FIT_RECEIPT.json").exists() or (original/"MODEL.pt").exists():
                raise ValueError("Original failed archive must not be eligible")
            substitute=a.recovery/archive_name("langmath101-posthoc",key)
            note=verify_recovery(substitute,key,hashlib.sha256(original_worker).hexdigest(),
                                 hashlib.sha256(effective_worker).hexdigest())
            seen_recovery.add(substitute.name)
            # Verification staging only: original recovery artifact is immutable,
            # not renamed, and not eligible for the original finalization.
            staging=a.output/"verification_staging"/archive_name("fit",key)
            shutil.copytree(substitute,staging)
            shutil.copy2(staging/"POSTHOC_MODEL.pt",staging/"MODEL.pt")
            shutil.copy2(staging/"POSTHOC_FIT_RECEIPT.json",staging/"FIT_RECEIPT.json")
            chosen=staging
        else:
            if (original/"FAILURE.json").exists():
                raise ValueError("New unregistered failure %s"%(key,))
            if not (original/"FIT_RECEIPT.json").is_file() or not (original/"MODEL.pt").is_file():
                raise ValueError("Original passing fit missing final receipt %s"%(key,))
            chosen=original
            note=None
        receipt=read(chosen/"FIT_RECEIPT.json")
        if (receipt["arm"],receipt["seed"],receipt["lane"])!=key:
            raise ValueError("Wrong final checkpoint identity")
        # Includes actual weights, AdamW moments, complete cold replay and
        # original source/recipe/initial tensor checks (frozen finalizer).
        item=F.verify_checkpoint(chosen,receipt,row,profile,manifest)
        item.pop("checkpoint_path",None)
        item.update(provenance="POSTHOC_TIMEOUT_AMENDMENT" if failure else "ORIGINAL_PREREGISTERED_PASS",
                    origin_run_id=RECOVERY_RUN if failure else ORIGINAL_RUN,
                    original_preregistered_fit_eligible=not failure)
        if note:
            item["amendment_sha256"]=sha(substitute/"POSTHOC_AMENDMENT.json")
            item["original_failed_checkpoint_sha256"]=sha(original/"FAILED_MODEL.pt")
        row["measured"]=receipt
        selected.append(item)
        verified.append({"arm":key[0],"seed":key[1],"lane":key[2],
                         "source_kind":item["provenance"],"checkpoint_sha256":item["checkpoint_sha256"],
                         "fit_receipt_sha256":item["fit_receipt_sha256"]})
        print(json.dumps({"checked":len(selected),"expected":138,"arm":key[0],
                          "seed":key[1],"lane":key[2],"provenance":item["provenance"]}),flush=True)
    originals={p.name for p in a.original.iterdir() if p.is_dir() and p.name.startswith("langmath101-fit-")}
    recoveries={p.name for p in a.recovery.iterdir() if p.is_dir() and p.name.startswith("langmath101-posthoc-")}
    if originals!=seen_original or recoveries!=seen_recovery:
        raise ValueError("Extra/missing original or exploratory recovery artifacts")
    if len(selected)!=138 or sum(not p["original_preregistered_fit_eligible"] for p in selected)!=3:
        raise ValueError("Mixed cohort cardinality differs")
    # This validates the measured operator/matched arithmetic convention, but
    # is not a preregistered-source/time gate and must never produce its seal.
    resource=gate.validate_program(full_rows,"final",root=C.ROOT)
    save(a.output/"EXPLORATORY_RESOURCE_GATE.json",{
        "status":"POSTHOC_INTERNAL_ARITHMETIC_GATE_NOT_ORIGINAL_PASS",
        "measured_gate":resource,
        "registered_wall_limit_seconds":2700,"recovered_wall_limit_seconds":4800,
        "original_cohort_completed_138":False,"architectural_superiority":False})
    save(a.output/"EXPLORATORY_VERIFIED_CHECKPOINTS.json",verified)
    save(a.output/"EXPLORATORY_PROVENANCE.json",{
        "status":"MIXED_COHORT_NOT_PREREGISTERED",
        "original_run_id":ORIGINAL_RUN,"recovery_run_id":RECOVERY_RUN,
        "original_valid_fits":135,"original_failed_fits":3,
        "recovery_valid_fits":3,
        "original_global_seal_completed":False,
        "original_DEV_scoring_permitted":False,
        "exploratory_DEV_scoring_permitted_after_this_barrier":True,
        "original_failed_keys":[list(k) for k in sorted(FAILURES)],
        "original_worker_sha256":hashlib.sha256(original_worker).hexdigest(),
        "posthoc_worker_sha256":hashlib.sha256(effective_worker).hexdigest(),
        "checkpoint_provenance":selected,
        "claim_limits":["NOT ORIGINAL PREREGISTERED SEAL","Mixed wall-clock budget",
                        "Partial dense arithmetic is not full FLOPs",
                        "No architecture superiority","Public exposed DEV only",
                        "No independent replication or protected TEST"]})
    gate_status={
        "schema":"langmath101.exploratory-seal.v1",
        "status":STATUS,"jobs":138,
        "checkpoint_count_original":135,"checkpoint_count_posthoc":3,
        "original_run_id":ORIGINAL_RUN,"recovery_run_id":RECOVERY_RUN,
        "source_hashes":profile["source_hashes"],
        "checkpoints":selected,
        "verified_checkpoints_sha256":sha(a.output/"EXPLORATORY_VERIFIED_CHECKPOINTS.json"),
        "resource_gate_sha256":sha(a.output/"EXPLORATORY_RESOURCE_GATE.json"),
        "provenance_sha256":sha(a.output/"EXPLORATORY_PROVENANCE.json"),
        "original_fit_eligible":False,"claim_architecture_superiority":False,
        "original_all_fit_seal":False,"no_DEV_data_read":True,
        "protected_TEST_access":False,
    }
    save(a.output/"EXPLORATORY_SEAL.json",gate_status)
    print(json.dumps({"status":STATUS,"checked":138,
                      "original":135,"posthoc":3}),flush=True)
    return gate_status

def gated_eval_source():
    src=(STUDY/"evaluate.py").read_text()
    start=src.index('    if not getattr(args, "fits_seal", None):')
    end=src.index('    payload = torch.load(args.checkpoint',start)
    old=src[start:end]
    if old.count('SEALED_ALL_FITS_BEFORE_DEV')!=1 or old.count('FINAL_OWNER_GATE.json')!=1:
        raise ValueError("Unexpected original DEV gate text")
    new='''    if not getattr(args, "fits_seal", None):
        raise ValueError("Exploratory seal required")
    fits_seal = json.loads(Path(args.fits_seal).read_text())
    if fits_seal.get("status") != "SEALED_EXPLORATORY_MIXED_PROVENANCE_BEFORE_DEV" or fits_seal.get("jobs") != 138:
        raise ValueError("Original seal cannot be substituted for exploratory seal")
    seal_dir = Path(args.fits_seal).parent
    for filename, field in (("EXPLORATORY_VERIFIED_CHECKPOINTS.json","verified_checkpoints_sha256"),
                            ("EXPLORATORY_RESOURCE_GATE.json","resource_gate_sha256"),
                            ("EXPLORATORY_PROVENANCE.json","provenance_sha256")):
        if sha(seal_dir/filename) != fits_seal[field]:
            raise ValueError("Exploratory evidence SHA mismatch: " + filename)
    lineage=json.loads((seal_dir/"EXPLORATORY_PROVENANCE.json").read_text())
    if (lineage.get("status") != "MIXED_COHORT_NOT_PREREGISTERED"
        or lineage.get("original_valid_fits") != 135 or lineage.get("original_failed_fits") != 3
        or lineage.get("recovery_valid_fits") != 3):
        raise ValueError("All checkpoint provenance required")
    if fits_seal.get("original_all_fit_seal") is not False or fits_seal.get("original_fit_eligible") is not False:
        raise ValueError("Original run may never be retroactively sealed")
    keys={(arm,seed,lane) for arm in C.arm_names() for seed in C.SEEDS
          for lane in ("parameter_exposure_matched","counted_arithmetic_matched")}
    entries=fits_seal.get("checkpoints",[])
    observed={(r.get("arm"),r.get("seed"),r.get("lane")) for r in entries}
    if len(entries)!=138 or len(observed)!=138 or observed!=keys:
        raise ValueError("All 138 mixed-provenance fit identities required")
    if sum(r.get("provenance")=="POSTHOC_TIMEOUT_AMENDMENT" for r in entries)!=3:
        raise ValueError("Exactly three amended fit sources required")
    if fits_seal.get("source_hashes")!=C.source_hashes():
        raise ValueError("Original executable source changed")
    sealed_matches=[r for r in entries if r.get("checkpoint_sha256")==actual_checkpoint_sha]
    if len(sealed_matches)!=1:
        raise ValueError("Checkpoint not included in unique exploratory seal")
'''
    effective=src[:start]+new+src[end:]
    if effective.replace(new,old)!=src:
        raise ValueError("Exploratory substitution modified evaluation calculations")
    compile(effective,str(STUDY/"evaluate.py"),"exec")
    return src,effective

def evaluate(a):
    from argparse import Namespace
    import torch
    raw,effective=gated_eval_source()
    seal_path=a.seal/"EXPLORATORY_SEAL.json"
    if not seal_path.is_file():
        raise ValueError("Missing audited mixed-provenance cohort seal")
    seal=read(seal_path)
    if seal.get("status")!=STATUS: raise ValueError("Wrong seal")
    candidates=[r for r in seal["checkpoints"] if
                (r["arm"],r["seed"],r["lane"])==(a.arm,a.seed,a.lane)]
    if len(candidates)!=1: raise ValueError("Unknown checkpoint identity")
    selection=candidates[0]
    if sha(a.checkpoint)!=selection["checkpoint_sha256"]:
        raise ValueError("Model checkpoint does not match approved lineage")
    context={"__name__":"langmath101_exploratory_evaluation",
             "__file__":str(STUDY/"evaluate.py")}
    exec(compile(effective,str(STUDY/"evaluate.py"),"exec"),context)
    context["run"](Namespace(checkpoint=a.checkpoint,sha256=selection["checkpoint_sha256"],
                    fits_seal=seal_path,fit=a.fit,data=a.data,output=a.output,
                    threads=2,conditions="full"))
    if not (a.output/"RESULTS.json").is_file():
        raise ValueError("Original model scoring did not complete")
    save(a.output/"EXPLORATORY_EVAL_PROVENANCE.json",{
        "status":"COMPLETED_MIXED_COHORT_EXPLORATORY_DEV",
        "original_evaluator_sha256":hashlib.sha256(raw.encode()).hexdigest(),
        "effective_evaluator_sha256":hashlib.sha256(effective.encode()).hexdigest(),
        "patch_scope":"ONLY original complete-138 seal gate, not predictions/scoring",
        "model_provenance":selection["provenance"],
        "origin_run_id":selection["origin_run_id"],
        "checkpoint_sha256":selection["checkpoint_sha256"],
        "seal_sha256":sha(seal_path),
        "DEV_classification":"Previously exposed public DEV - internal exploratory",
        "original_run_pass":False,
        "eligible_for_architecture_superiority":False,
        "protected_TEST_access":False})
    print(json.dumps({"status":"EXPLORATORY_DEV_EVALUATED",
                      "arm":a.arm,"seed":a.seed,"lane":a.lane,
                      "source":selection["provenance"]}),flush=True)

def aggregate(a):
    # The original statistical engine is unchanged; output names and status
    # explicitly distinguish the mixed posthoc cohort from original study.
    from argparse import Namespace
    manifest=STUDY/"FAIR_COMPARISON.json"
    cohorts=identities(read(manifest))
    if not a.seal.joinpath("EXPLORATORY_SEAL.json").is_file():
        raise ValueError("Mixed-provenance seal required")
    seal=read(a.seal/"EXPLORATORY_SEAL.json")
    if seal.get("status")!=STATUS: raise ValueError("Wrong exploratory seal")
    expected={(r["arm"],r["seed"],r["lane"]):r for r in seal["checkpoints"]}
    seen=set()
    for path in a.evaluations.rglob("EXPLORATORY_EVAL_PROVENANCE.json"):
        value=read(path)
        results=read(path.parent/"RESULTS.json")
        key=(results["arm"],results["seed"],results["lane"])
        if key not in cohorts or key in seen:raise ValueError("Extra or duplicate DEV evaluation")
        if (value.get("checkpoint_sha256")!=expected[key]["checkpoint_sha256"]
           or value.get("model_provenance")!=expected[key]["provenance"]
           or value.get("seal_sha256")!=sha(a.seal/"EXPLORATORY_SEAL.json")):
            raise ValueError("Evaluation/source provenance mismatch")
        seen.add(key)
    if len(seen)!=138 or seen!=set(cohorts):
        raise ValueError("Require all 138 evaluations, not a selected subset")
    agg=mod("langmath101_exploratory_aggregate",STUDY/"aggregate.py")
    a.output.mkdir(parents=True,exist_ok=False)
    inner=a.output/"original_stats_unqualified"
    exit_code=agg.aggregate(Namespace(manifest=manifest,evaluations=a.evaluations,output=inner))
    if exit_code!=0:raise ValueError("Original statistical engine reports invalid/incomplete cohort")
    report=read(inner/"REPORT.json")
    if report["valid_evaluations"]!=138:
        raise ValueError("All genuine-source DEV evaluations required")
    report["status"]="COMPLETE_EXPLORATORY_MIXED_PROVENANCE_NOT_ORIGINAL_STUDY"
    report["original_statistical_output_sha256"]=sha(inner/"REPORT.json")
    report["mixed_cohort_seal_sha256"]=sha(a.seal/"EXPLORATORY_SEAL.json")
    report["original_preregistered_result"]="INCOMPLETE_135_OF_138_FITS"
    report["posthoc_recovery"]="3_OF_3_TRAINED_EXPLORATORY_ONLY"
    report["original_all_fit_seal"]=False
    report["training_wall_time_matching"]="NOT_ESTABLISHED_FOR_POSTHOC_RECOVERIES"
    report["superiority"]="NOT_ESTABLISHED"
    save(a.output/"EXPLORATORY_REPORT.json",report)
    (a.output/"EXPLORATORY_REPORT.md").write_text(
        "# LANGMATH101 exploratory mixed-provenance DEV analysis\n\n"
        "NOT an original preregistered result. 135 original passing fits; "
        "three separately time-amended posthoc recoveries. Public exposed DEV, "
        "no protected replication. Original study failed its 138-fit seal.\n\n"
        "Raw exploratory descriptive and paired results are in EXPLORATORY_REPORT.json "
        "and original_stats_unqualified/. Do not claim overall model superiority.\n")
    print(json.dumps({"status":report["status"],"valid":138}),flush=True)

def main():
    p=argparse.ArgumentParser()
    cmd=p.add_subparsers(dest="cmd",required=True)
    s=cmd.add_parser("self-test")
    seal_p=cmd.add_parser("seal")
    seal_p.add_argument("--original",type=Path,required=True)
    seal_p.add_argument("--recovery",type=Path,required=True)
    seal_p.add_argument("--output",type=Path,required=True)
    ev=cmd.add_parser("evaluate")
    for name in ("arm","lane"):ev.add_argument("--"+name,required=True)
    ev.add_argument("--seed",type=int,required=True)
    for name in ("checkpoint","seal","fit","data","output"):
        ev.add_argument("--"+name,type=Path,required=True)
    ag=cmd.add_parser("aggregate")
    for name in ("evaluations","seal","output"):
        ag.add_argument("--"+name,type=Path,required=True)
    a=p.parse_args()
    if a.cmd=="self-test":
        original,effective=gated_eval_source()
        amended_script().effective_source()
        assert "SEALED_EXPLORATORY_MIXED_PROVENANCE_BEFORE_DEV" in effective
        assert "SEALED_ALL_FITS_BEFORE_DEV" not in effective
        print(json.dumps({"status":"EXPLORATORY_GATE_ISOLATED",
                          "original_evaluator_sha256":hashlib.sha256(original.encode()).hexdigest(),
                          "effective_evaluator_sha256":hashlib.sha256(effective.encode()).hexdigest(),
                          "original_study_unchanged":True}),flush=True)
    elif a.cmd=="seal":seal(a)
    elif a.cmd=="evaluate":evaluate(a)
    else:aggregate(a)

if __name__=="__main__":main()
