#!/usr/bin/env python3
"""Replay original real Backblaze classical risk checkpoint WITHOUT fitting.

Compares exact FIT-cell counts and complete original exposed DEV score receipt
from original source against Git-owned coefficients. Protected data unopened.
"""
import hashlib
import json
from pathlib import Path
import sys

import model02_backblaze_exploratory as exp

ROOT=Path(__file__).resolve().parents[1]
BUNDLE=ROOT/"research/replay-bundles/PRODUCT-MODEL02-BACKBLAZE-CLASSICAL"
MODEL=BUNDLE/"ORIGINAL_CHECKPOINT.json"
RESULT=BUNDLE/"ORIGINAL_EXPOSED_DEV.json"

def replay(data):
    model=json.loads(MODEL.read_bytes())
    original=json.loads(RESULT.read_bytes())
    if (model.get("schema")!="model02-classical-smart-binary-original-v1"
        or model.get("source_zip_sha256")!=exp.ORIGINAL_SHA or
        original.get("source_zip_sha256")!=exp.ORIGINAL_SHA):
        raise exp.EvidenceError("original Git checkpoint/score source contract mismatch")
    timeline=exp.records_from_original(data)
    fit,fi=exp.case_rows(timeline,exp.period.FIRST,exp.period.FIT_END)
    dev,di=exp.case_rows(timeline,exp.period.FIT_END,exp.period.EXPOSED_END)
    if (model["fit_windows"]!=sum(n for p,n in fit.values())
        or model["fit_positive_windows"]!=sum(p for p,n in fit.values())
        or any(model["cells"][k]["n"]!=n or model["cells"][k]["pos"]!=p
               for k,(p,n) in fit.items())):
        raise exp.EvidenceError("frozen trained bin counts do not match FIT only")
    report={
      "schema":"model02-real-exposed-risk-development-v1",
      "source_zip_sha256":exp.ORIGINAL_SHA,
      "FIT":{**fi,**exp.score(model,fit,fi["distinct_drives_with_positive_windows"])},
      "DEV":{**di,**exp.score(model,dev,di["distinct_drives_with_positive_windows"])},
      "trained_classical_reference":True,"trained_TAC_model":False,
      "independent_evaluation":False,"protected_future_member_contents_opened":False,
      "scientific_qualification":"NOT_QUALIFIED",
      "warnings":original["warnings"],
    }
    if report!=original:
        raise exp.EvidenceError("original DEV receipt is not exactly replayed")
    return {"schema":"model02-original-source-no-fit-replay-v1",
            "source_zip_sha256":exp.ORIGINAL_SHA,
            "checkpoint_file_sha256":hashlib.sha256(MODEL.read_bytes()).hexdigest(),
            "source_real_exposed_dev_windows":di["matured_windows"],
            "original_dev_brier":report["DEV"]["brier"],
            "original_dev_log_loss":report["DEV"]["log_loss"],
            "original_score_exactly_replayed":True,
            "trained_during_replay":False,
            "independent_qualification":"NOT_ESTABLISHED"}

if __name__=="__main__":
    try:
        print(exp.canonical(replay(exp.pub.fetch_zip())))
    except (exp.EvidenceError,OSError,ValueError,TypeError) as exc:
        print(exp.canonical({"status":"BLOCKED_OR_FAILED","reason":str(exc)}),file=sys.stderr)
        sys.exit(2)
