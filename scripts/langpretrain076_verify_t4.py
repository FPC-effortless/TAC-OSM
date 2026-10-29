#!/usr/bin/env python3
"""Verify actual downloaded dual-T4 FIT optimizer receipts, without executing GPUs.

Output is fail-closed. Kaggle CLI returning success is NOT a completed model run.
Requires two original worker JSON results, actual checkpoint bytes and runtime
report. No optimizer or model load is performed by this verifier.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path

SHA = "dd5aabf90646be409e3f18ecf763dbcefc84b6575e12ff54f4a7195f92ff150e"
SEEDS = (1701,1702)
EXPECTED_RUN_ID = "langpretrain076-h48-free-t4"
WORKLOAD_PATH = Path(__file__).resolve().parents[1] / "scripts/langpretrain076_t4_measure.py"
TIMED_BYTES = 32768
TOTAL_BYTES = 40960
PARAMS = 499623


class ReceiptError(ValueError):
    pass


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(directory: Path) -> dict:
    root=directory.resolve()
    runtime_file=root/"phone-research-runtime.json"
    if not runtime_file.is_file():
        raise ReceiptError("Kaggle did not provide actual two-T4 runtime evidence")
    runtime=json.loads(runtime_file.read_text())
    if (runtime.get("schema")!="tac-osm.kaggle-t4x2-runtime.v1"
        or runtime.get("run_id") != EXPECTED_RUN_ID
        or runtime.get("workload_sha256") != sha(WORKLOAD_PATH)
        or runtime.get("requested_accelerator")!="NvidiaTeslaT4"
        or runtime.get("status")!="COMPLETED_UNQUALIFIED_REQUIRES_INDEPENDENT_EVALUATION"
        or runtime.get("seeds")!=list(SEEDS) or runtime.get("strategy")!="seed_shard"):
        raise ReceiptError("Kaggle runtime failed or wrong source-bound experiment")
    obs=runtime.get("observed",{})
    if (obs.get("count") != 2 or obs.get("accelerator") != "NvidiaTeslaT4"
        or len(obs.get("names",[]))!=2
        or any("T4" not in str(x).upper() for x in obs["names"])):
        raise ReceiptError("two actual self-reported T4 devices not established")
    results=runtime.get("results")
    if not isinstance(results,list) or len(results)!=2 or {r.get("worker_index") for r in results}!={0,1} or any(r.get("exit_code")!=0 for r in results):
        raise ReceiptError("two workers did not both finish successfully")
    if not isinstance(runtime.get("elapsed_seconds"),int) or runtime["elapsed_seconds"]<=0:
        raise ReceiptError("not enough verified runtime wall information")
    rows=[]
    for slot,seed in enumerate(SEEDS):
        f=root/f"seed-{seed}.json"
        cp=root/f"seed-{seed}.pt"
        if not f.is_file() or not cp.is_file():
            raise ReceiptError("actual seed evidence or trained checkpoint missing")
        d=json.loads(f.read_text())
        if (d.get("schema")!="tac-osm.langpretrain076.t4-throughput.v1"
            or d.get("study")!="LANGPRETRAIN076"
            or d.get("seed")!=seed or d.get("physical_slot")!=slot
            or "T4" not in str(d.get("device_name","")).upper()
            or d.get("device_name") != obs["names"][slot]
            or d.get("source_sha256")!=SHA
            or d.get("trainable_parameters")!=PARAMS
            or d.get("deployed_parameters")!=PARAMS
            or d.get("model_tier")!="h48"
            or d.get("warmup_updates")!=8 or d.get("measured_updates")!=32
            or d.get("training_target_bytes")!=TOTAL_BYTES
            or d.get("measured_target_bytes")!=TIMED_BYTES
            or d.get("batch")!=8 or d.get("context_bytes")!=128
            or d.get("checkpoint_cold_reload_logit_parity")!="PASS_ATOL1E-6_RTOL1E-6_FIT_ONLY"
            or d.get("DEV")!="NOT_ACCESSED"
            or d.get("protected_TEST")!="NOT_ACCESSED"
            or d.get("paid_cloud_spend_authorized_usd")!=0
            or sha(cp)!=d.get("checkpoint_sha256")):
            raise ReceiptError(f"seed {seed} model/source/checkpoint contract mismatch")
        duration=d.get("measured_synced_wall_seconds")
        reported=d.get("measured_byte_tokens_per_second")
        memory=d.get("peak_cuda_tensor_allocated_bytes")
        if (type(duration) not in (int,float) or not math.isfinite(duration) or duration<=0
            or type(reported) not in (int,float) or not math.isfinite(reported) or reported<=0
            or abs(TIMED_BYTES/duration-reported)/reported>1e-9
            or type(memory) is not int or memory<=0):
            raise ReceiptError("invalid GPU-measured duration, raw byte rate or VRAM")
        rows.append({"seed":seed,"physical_slot":slot,"device":d["device_name"],
                     "measured_raw_byte_tokens_per_second":reported,
                     "synced_wall_seconds":duration,
                     "peak_cuda_tensor_allocated_bytes":memory,
                     "checkpoint_sha256":d["checkpoint_sha256"]})
    # Exact runtime wall duration includes initialization/worker orchestration.
    # It is intentionally NOT represented as pure training-throughput aggregate
    # because the two worker timed intervals have no shared start/end clocks.
    return {"schema":"tac-osm.langpretrain076-verified-throughput.v1",
            "status":"SELF_REPORTED_DUAL_T4_TRAIN_RECEIPTS_CHECKED",
            "scientific_status":"INTERNAL_EFFICIENCY_ONLY_NOT_INDEPENDENT",
            "original_source_sha256":SHA,
            "workers":rows,
            "combined_training_target_exposures":2*TOTAL_BYTES,
            "combined_timed_target_exposures":2*TIMED_BYTES,
            "runtime_elapsed_seconds":runtime["elapsed_seconds"],
            "aggregate_timed_tokens_per_second":"NOT_COMPARABLE_UNTIL_ALIGNED_WORKER_CLOCKS",
            "source_quality_dev":"NOT_EVALUATED",
            "protected_TEST":"NOT_ACCESSED",
            "paid_compute_claim":"NOT_VERIFIED_BY_RESULT_ONLY",
            "Kaggle_execution_receipt_trust":"SELF_REPORTED_NOT_HARDWARE_ATTESTED"}


def main() -> int:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--outputs",required=True,type=Path)
    p.add_argument("--report",required=True,type=Path)
    a=p.parse_args()
    try:
        result=verify(a.outputs)
    except (ReceiptError,OSError,ValueError,TypeError,KeyError) as exc:
        p.exit(2,f"BLOCKED_UNVERIFIED_GPU_RESULT: {type(exc).__name__}: {exc}\n")
    if a.report.exists(): p.exit(2,"refuse overwrite real GPU evidence\n")
    a.report.parent.mkdir(parents=True,exist_ok=True)
    a.report.write_text(json.dumps(result,sort_keys=True,indent=2,allow_nan=False)+"\n")
    print(json.dumps({"status":result["status"],
                      "seed_rates":{str(x["seed"]):x["measured_raw_byte_tokens_per_second"] for x in result["workers"]}}))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
