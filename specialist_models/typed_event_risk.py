#!/usr/bin/env python3
"""Independent classical binary event-risk baseline (NOT TAC-OSM or a qualified product).
Uses explicitly source-timestamped observations; fit, DEV, inference are separate.
"""
from __future__ import annotations
import argparse
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import sys

class EvidenceError(ValueError):
    pass

def canonical(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False)

def utc(value):
    if not isinstance(value, str):
        raise EvidenceError("UTC ISO timestamp required")
    try:
        t = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise EvidenceError("invalid ISO time") from exc
    if t.tzinfo is None or t.utcoffset().total_seconds() != 0:
        raise EvidenceError("timestamps must be UTC-aware")
    return t

def source_hash(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def read_rows(path):
    rows = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            if len(rows) >= 1000000:
                raise EvidenceError("dataset exceeds bounded intake")
            rows.append(json.loads(line))
    if not rows:
        raise EvidenceError("dataset is empty")
    return rows

def parse_observation(row, features, label=False, cutoff=None):
    if not isinstance(row, dict):
        raise EvidenceError("observation must be object")
    expected = {"observed_at", "features"} | ({"outcome", "outcome_at"} if label else set())
    if set(row) != expected:
        raise EvidenceError("unexpected/hidden source fields or labels rejected")
    t = utc(row.get("observed_at"))
    if cutoff and not t < cutoff:
        raise EvidenceError("FIT observation must precede source cutoff")
    fields = row.get("features")
    if not isinstance(fields, dict) or set(fields) != set(features):
        raise EvidenceError("exact feature schema is required")
    vals = []
    for name in features:
        entry = fields[name]
        if not isinstance(entry, dict):
            raise EvidenceError("feature requires value and available_at")
        if utc(entry.get("available_at")) > t:
            raise EvidenceError("future feature detected")
        v = entry.get("value")
        if type(v) not in (int, float) or not math.isfinite(v):
            raise EvidenceError("feature must be finite numeric")
        vals.append(float(v))
    if label:
        y = row.get("outcome")
        if type(y) is not int or y not in (0, 1):
            raise EvidenceError("binary outcome required")
        if utc(row.get("outcome_at")) < t:
            raise EvidenceError("outcome predates observation")
        if cutoff and not utc(row["outcome_at"]) < cutoff:
            raise EvidenceError("training outcome not yet source-available at cutoff")
        return vals, y
    if "outcome" in row or "outcome_at" in row:
        raise EvidenceError("labels must not enter inference")
    return vals

def sigmoid(z):
    if z >= 0:
        return 1. / (1. + math.exp(-z))
    e = math.exp(z)
    return e / (1. + e)

def fit(rows, source, actual_source_hash):
    names = source.get("features")
    if (not isinstance(names, list) or not 1 <= len(names) <= 32 or
        any(not isinstance(x, str) or not x for x in names) or len(set(names)) != len(names)):
        raise EvidenceError("1 to 32 named numeric features required")
    if source.get("status") != "source-claimed-not-independently-audited":
        raise EvidenceError("source status may not claim independent qualification")
    if not isinstance(source.get("source_url"), str) or not source["source_url"].startswith(("https://","http://")):
        raise EvidenceError("source URL is required")
    if not source.get("dataset_license") or not source.get("data_steward"):
        raise EvidenceError("data license and steward declarations required")
    cutoff = utc(source.get("fit_cutoff"))
    xy = [parse_observation(row, names, True, cutoff) for row in rows]
    xs = [x for x, y in xy]; labels = [y for x, y in xy]
    if set(labels) != {0, 1}:
        raise EvidenceError("both training classes required")
    n, d = len(xs), len(names)
    mean = [sum(x[j] for x in xs) / n for j in range(d)]
    std = [max(math.sqrt(sum((x[j]-mean[j])**2 for x in xs)/n),1e-8) for j in range(d)]
    standardized = [[max(-20., min(20., (x[j]-mean[j])/std[j])) for j in range(d)] for x in xs]
    w = [0.] * (d + 1)
    for _ in range(300):
        grad = [0.] * (d + 1)
        for x,y in zip(standardized,labels):
            error = sigmoid(w[0] + sum(w[j+1]*v for j,v in enumerate(x))) - y
            grad[0] += error
            for j,v in enumerate(x):
                grad[j+1] += error*v
        w[0] -= .15 * grad[0]/n
        for j in range(d):
            w[j+1] -= .15 * (grad[j+1]/n + .005*w[j+1])
    return {"schema": "typed-event-risk-classical-v1",
            "qualification": "DEVELOPMENT_ONLY_NOT_CALIBRATED",
            "features":names,"mean":mean,"std":std,"weights":w,
            "fit_cutoff":source["fit_cutoff"],
            "fit_source_sha256":actual_source_hash,
            "source_url":source["source_url"],"license_claim":source["dataset_license"],
            "steward_claim":source["data_steward"],"fit_cases":n,
            "fit_positives":sum(labels),"algorithm":"l2-logistic-300-full-batch"}

def predict(model, observation):
    x = parse_observation(observation, model["features"])
    z = [(max(-20., min(20., (x[j]-model["mean"][j])/model["std"][j])))
         for j in range(len(x))]
    w = model["weights"]
    prob = sigmoid(w[0] + sum(w[j+1]*v for j,v in enumerate(z)))
    return {"output_type":"binary-probability", "probability":prob,
            "predicted_label":int(prob >= .5),
            "source_observed_at":observation["observed_at"],
            "calibration":"NOT_INDEPENDENTLY_QUALIFIED", "authority":"ADVISORY_ONLY"}

def evaluate(model, rows):
    cutoff = utc(model["fit_cutoff"])
    predicted, labels = [], []
    for row in rows:
        if utc(row.get("observed_at")) < cutoff:
            raise EvidenceError("DEV overlaps FIT cutoff")
        y = row.get("outcome")
        if type(y) is not int or y not in (0,1):
            raise EvidenceError("evaluator requires binary label")
        if utc(row.get("outcome_at")) < utc(row["observed_at"]):
            raise EvidenceError("DEV outcome predates observed state")
        no_label = {k:v for k,v in row.items() if k not in ("outcome","outcome_at")}
        predicted.append(predict(model,no_label)["probability"])
        labels.append(y)
    n=len(rows)
    baseline=sum(labels)/n
    return {"qualification":"DEV_ONLY_NOT_INDEPENDENT_TEST","cases":n,
            "brier":sum((p-y)**2 for p,y in zip(predicted,labels))/n,
            "log_loss":-sum(y*math.log(max(1e-12,p))+(1-y)*math.log(max(1e-12,1-p))
                            for p,y in zip(predicted,labels))/n,
            "dev_positive_rate":baseline,
            "dev_prevalence_reference_brier_NONPROSPECTIVE":
                sum((baseline-y)**2 for y in labels)/n}

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest="cmd",required=True)
    a=sub.add_parser("fit"); a.add_argument("--fit-cases",type=Path,required=True)
    a.add_argument("--source",type=Path,required=True); a.add_argument("--out",type=Path,required=True)
    b=sub.add_parser("evaluate"); b.add_argument("--model",type=Path,required=True)
    b.add_argument("--dev-cases",type=Path,required=True)
    c=sub.add_parser("infer"); c.add_argument("--model",type=Path,required=True)
    c.add_argument("--observation",type=Path,required=True)
    args=p.parse_args(argv)
    try:
        if args.cmd=="fit":
            if args.out.exists():
                raise EvidenceError("refusing overwrite of model")
            model=fit(read_rows(args.fit_cases),
                      json.loads(args.source.read_text()),source_hash(args.fit_cases))
            with args.out.open("x",encoding="utf-8") as f:
                json.dump(model,f,sort_keys=True,indent=2)
            output={"qualification":"DEVELOPMENT_ONLY_NOT_CALIBRATED",
                    "fit_cases":model["fit_cases"],"source_sha256":model["fit_source_sha256"]}
        else:
            model=json.loads(args.model.read_text())
            if model.get("schema")!="typed-event-risk-classical-v1":
                raise EvidenceError("model schema invalid")
            if args.cmd=="evaluate":
                output=evaluate(model,read_rows(args.dev_cases))
            else:
                output=predict(model,json.loads(args.observation.read_text()))
        print(canonical(output))
        return 0
    except (EvidenceError,ValueError,OSError,TypeError,KeyError) as exc:
        print(canonical({"status":"BLOCKED_OR_FAILED","reason":str(exc)}),file=sys.stderr)
        return 2

if __name__ == "__main__":
    raise SystemExit(main())
