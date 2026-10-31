"""Post-hoc, fail-closed LANGMATH101 timeout recovery; never upgrades original evidence.

This executes a source-verified copy of the frozen worker with EXACTLY ONE
time-limit predicate changed. The effective source SHA is preserved in the
recovery artifact. No architecture, weights, data, losses, optimizer, stopping
target, or counted-work ledger is changed. Results are never eligible for the
original preregistered seal and cannot silently enter original DEV reports.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
STUDY = ROOT / "research/mathematical-engineering/langmath101"
SOURCE = STUDY / "train.py"
MANIFEST = STUDY / "FAIR_COMPARISON.json"
FROZEN_RUN = 41174892770
REGISTERED_LIMIT = 2700
AMENDED_LIMIT = 4800
FAILURES = {
    ("a5_adaptive", 10102, "counted_arithmetic_matched"),
    ("a5_adaptive", 10103, "counted_arithmetic_matched"),
    ("a5_adaptive", 10103, "parameter_exposure_matched"),
}
ORIGINAL = (
    "if time.perf_counter()-started>C.RECIPE['max_arm_seconds']:"
    "raise TimeoutError('Preregistered per-fit CPU budget exhausted')"
)
AMENDED = (
    "if time.perf_counter()-started>POSTHOC_TIMEOUT_SECONDS:"
    "raise TimeoutError('Posthoc amended 4800-second wall budget exhausted')"
)

def sha(data):
    return hashlib.sha256(data).hexdigest()

def effective_source():
    manifest = json.loads(MANIFEST.read_text())
    profile = json.loads((ROOT / manifest["operator_profile_path"]).read_text())
    key = SOURCE.relative_to(ROOT).as_posix()
    raw = SOURCE.read_bytes()
    assert sha(raw) == profile["source_hashes"][key], "Frozen worker source changed"
    assert profile["training_recipe"]["max_arm_seconds"] == REGISTERED_LIMIT
    assert raw.count(ORIGINAL.encode()) == 1, "Timeout marker must be unique"
    effective = raw.replace(ORIGINAL.encode(), AMENDED.encode())
    assert effective.count(AMENDED.encode()) == 1
    # Explicitly prove the only difference in executed source is the predicate.
    assert effective.replace(AMENDED.encode(), ORIGINAL.encode()) == raw
    compile(effective, str(SOURCE), "exec")
    return raw, effective, manifest, profile

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--self-test", action="store_true")
    p.add_argument("--fit", type=Path)
    p.add_argument("--tokenizers", type=Path)
    p.add_argument("--initial", type=Path)
    p.add_argument("--arm")
    p.add_argument("--seed", type=int)
    p.add_argument("--lane")
    p.add_argument("--output", type=Path)
    a = p.parse_args()
    raw, effective, manifest, profile = effective_source()
    if a.self_test:
        print(json.dumps({"status":"PASS_SOURCE_ONLY_TIME_LIMIT_CHANGED",
                          "original_worker_sha256":sha(raw),
                          "effective_worker_sha256":sha(effective),
                          "registered_limit_seconds":REGISTERED_LIMIT,
                          "posthoc_limit_seconds":AMENDED_LIMIT}))
        return
    target = (a.arm, a.seed, a.lane)
    if target not in FAILURES:
        p.error("Posthoc recovery is restricted to the three confirmed failed fits")
    if not all((a.fit, a.tokenizers, a.initial, a.output)):
        p.error("Original FIT, tokenizer, initial checkpoint and output paths required")
    if a.output.exists():
        raise FileExistsError("Recovery outputs are append-only; choose an unused path")
    assert any((r["architecture"],r["seed"],r["lane"]) == target
               for r in manifest["arms"]), "Unregistered fit row"
    context = {"__name__":"langmath101_posthoc_worker",
               "__file__":str(SOURCE),
               "POSTHOC_TIMEOUT_SECONDS":AMENDED_LIMIT}
    note = {
        "study":"LANGMATH101_POSTHOC_TIMEOUT_AMENDMENT",
        "original_run_id":FROZEN_RUN,
        "original_registered_limit_seconds":REGISTERED_LIMIT,
        "posthoc_limit_seconds":AMENDED_LIMIT,
        "original_worker_sha256":sha(raw),
        "effective_worker_sha256":sha(effective),
        "exact_patch":{"from":ORIGINAL,"to":AMENDED},
        "architecture":a.arm,"seed":a.seed,"lane":a.lane,
        "status":"POSTHOC_NOT_PREREGISTERED_NOT_QUALIFIED",
        "DEV_access":False,"protected_TEST_access":False,
        "original_fit_eligibility":False,
        "reason":"Original job exceeded preregistered wall budget; follow-up "
                 "diagnostic changes only the wall ceiling, after observing failure.",
    }
    try:
        exec(compile(effective, str(SOURCE), "exec"), context)
        from argparse import Namespace
        context["train"](Namespace(
            manifest=MANIFEST, fit=a.fit, tokenizers=a.tokenizers,
            initial=a.initial, arm=a.arm, seed=a.seed, lane=a.lane,
            output=a.output))
    except BaseException:
        if a.output.exists():
            (a.output / "POSTHOC_AMENDMENT.json").write_text(
                json.dumps({**note,"outcome":"FAILED"},indent=2)+"\n")
        raise
    receipt_path=a.output/"FIT_RECEIPT.json"
    checkpoint_path=a.output/"MODEL.pt"
    if not receipt_path.is_file() or not checkpoint_path.is_file():
        raise RuntimeError("Required sealed diagnostic checkpoint missing")
    original_receipt=json.loads(receipt_path.read_text())
    assert original_receipt["status"]=="SEALED_BEFORE_DEV"
    note["outcome"]="TRAINING_COMPLETED_EXPLORATORY_ONLY"
    note["actual_steps"]=original_receipt["actual_steps"]
    note["elapsed_seconds"]=original_receipt["elapsed_seconds"]
    note["original_checkpoint_sha256"]=sha(checkpoint_path.read_bytes())
    receipt_path.rename(a.output/"POSTHOC_FIT_RECEIPT.json")
    checkpoint_path.rename(a.output/"POSTHOC_MODEL.pt")
    (a.output / "POSTHOC_AMENDMENT.json").write_text(json.dumps(note,indent=2)+"\n")
    assert not (a.output/"MODEL.pt").exists()
    assert not (a.output/"FIT_RECEIPT.json").exists()
    print(json.dumps({"status":"POSTHOC_RECOVERY_COMPLETED_NOT_PREREGISTERED",
                      "arm":a.arm,"seed":a.seed,"lane":a.lane,
                      "actual_steps":note["actual_steps"]}),flush=True)

if __name__=="__main__":
    main()
