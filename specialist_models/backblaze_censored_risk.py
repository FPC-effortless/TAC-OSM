#!/usr/bin/env python3
"""Fail-closed, right-censoring-aware Backblaze daily CSV *development* adapter.

No public dataset downloads, hidden TEST labels, neural fitting or trained
risk/censoring claims. Creates input rows for typed_event_risk.py only where
a known publisher-recorded event or a completely observed horizon exists.
"""
from __future__ import annotations

import argparse
import csv
from datetime import date, datetime, timedelta, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import sys

OFFICIAL_SOURCE = "https://www.backblaze.com/cloud-storage/resources/hard-drive-test-data"
SMART_DEFAULT = ("smart_5_raw", "smart_9_raw", "smart_187_raw",
                 "smart_197_raw", "smart_198_raw")


class SourceError(ValueError):
    pass


def day(value):
    if not isinstance(value, str) or re.fullmatch(r"\d{4}-\d{2}-\d{2}", value) is None:
        raise SourceError("YYYY-MM-DD dates only")
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise SourceError("invalid date") from exc


def end_of_day(d):
    return datetime(d.year, d.month, d.day, 23, 59, 59, tzinfo=timezone.utc).isoformat()


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def config_check(manifest, phase):
    if not isinstance(manifest, dict) or manifest.get("schema") != "backblaze-entity-cohort-v1":
        raise SourceError("unsupported cohort manifest")
    if manifest.get("source_url") != OFFICIAL_SOURCE:
        raise SourceError("source must identify official Backblaze publisher")
    if manifest.get("permission") != "review-required-no-redistribution":
        raise SourceError("explicit no-redistribution/review declaration required")
    enrollment = day(manifest.get("enrollment_date"))
    fit_end = day(manifest.get("fit_end"))
    dev_end = day(manifest.get("dev_end"))
    if not enrollment < fit_end < dev_end:
        raise SourceError("chronological enrollment/FIT/DEV cutoffs required")
    horizon = manifest.get("horizon_days")
    if type(horizon) is not int or not 1 <= horizon <= 60:
        raise SourceError("invalid positive horizon")
    features = manifest.get("smart_fields")
    if not isinstance(features, list) or not features or len(set(features)) != len(features):
        raise SourceError("explicit distinct smart_fields required")
    if any(not isinstance(f, str) or re.fullmatch(r"smart_\d+_raw", f) is None for f in features):
        raise SourceError("only enumerated raw SMART attribute names supported")
    cohort = manifest.get("enrolled_drives")
    if not isinstance(cohort, list) or not 1 <= len(cohort) <= 2000:
        raise SourceError("frozen, bounded original entity cohort required")
    identities = set()
    for x in cohort:
        if not isinstance(x, dict) or set(x) != {"serial_number", "model"}:
            raise SourceError("cohort needs source-owned serial_number and model")
        pair = (x["serial_number"], x["model"])
        if any(not isinstance(v, str) or not v or len(v) > 200 for v in pair):
            raise SourceError("invalid source identity")
        if pair in identities:
            raise SourceError("duplicate enrolled identity")
        identities.add(pair)
    files = manifest.get("csv_files")
    if not isinstance(files, list) or not files or len(files) > 400:
        raise SourceError("bounded daily source files required")
    expected_end = fit_end if phase == "fit" else dev_end
    expected_dates = [enrollment + timedelta(days=i) for i in range((expected_end - enrollment).days)]
    listed = []
    for x in files:
        if not isinstance(x, dict) or set(x) != {"date", "path", "sha256"}:
            raise SourceError("each daily source file needs date/path/sha256")
        d = day(x["date"])
        if d >= dev_end:
            raise SourceError("protected future period included in source manifest")
        if x["path"] != d.isoformat() + ".csv":
            raise SourceError("only daily date-stamped CSV files allowed")
        if not isinstance(x["sha256"], str) or re.fullmatch("[0-9a-f]{64}", x["sha256"]) is None:
            raise SourceError("original CSV SHA256 required")
        listed.append(d)
    if listed != sorted(set(listed)):
        raise SourceError("daily source files must be unique and ordered")
    if phase == "fit":
        requested = [d for d in listed if d < fit_end]
    elif phase == "dev":
        requested = listed
    else:
        raise SourceError("only FIT or exposed DEV supported; protected TEST unavailable")
    if requested != expected_dates:
        raise SourceError("missing daily source files in declared cohort period")
    return manifest, identities, expected_end


def load_selected(manifest, phase, csv_root):
    manifest, selected, boundary = config_check(manifest, phase)
    records = {ident: {} for ident in selected}
    seen_first = set()
    total = 0
    for item in manifest["csv_files"]:
        d = day(item["date"])
        if d >= boundary:
            continue
        root = csv_root.resolve()
        path = root / item["path"]
        if path.is_symlink() or not path.resolve().is_relative_to(root) or not path.is_file():
            raise SourceError("missing/out-of-root/symlink source CSV")
        if sha256(path) != item["sha256"]:
            raise SourceError("raw publisher CSV file hash mismatch")
        with path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            required = {"date", "serial_number", "model", "failure", *manifest["smart_fields"]}
            if not required.issubset(set(reader.fieldnames or [])):
                raise SourceError("source schema changed or required SMART attribute missing")
            for row in reader:
                identity = (row["serial_number"], row["model"])
                if identity not in selected:
                    continue
                if row["date"] != d.isoformat():
                    raise SourceError("CSV row source day mismatch")
                if row["failure"] not in ("0", "1"):
                    raise SourceError("invalid failure flag")
                if d in records[identity]:
                    raise SourceError("duplicate daily source-owned drive identity")
                smarts = {}
                for field in manifest["smart_fields"]:
                    raw = row[field]
                    if raw is None or raw == "":
                        smarts[field] = None
                    else:
                        try:
                            val = float(raw)
                        except ValueError as exc:
                            raise SourceError("invalid SMART value") from exc
                        if not math.isfinite(val) or val < 0:
                            raise SourceError("invalid negative/nonfinite SMART raw")
                        smarts[field] = val
                records[identity][d] = {"failed": row["failure"] == "1", "smart": smarts}
                if d == day(manifest["enrollment_date"]) and row["failure"] == "0":
                    seen_first.add(identity)
                total += 1
    if seen_first != selected:
        raise SourceError("all members must be present and functioning on enrollment date")
    for ident, timeline in records.items():
        failure_days = [d for d, r in timeline.items() if r["failed"]]
        if len(failure_days) > 1 or (failure_days and max(timeline) > failure_days[0]):
            raise SourceError("multiple or resurrected drive failure identities")
    return records, total


def features_for(timeline, d, smart_fields):
    features = {}
    current = timeline[d]["smart"]
    for f in smart_fields:
        value = current[f]
        amount = math.log1p(value) if value is not None else 0.
        features[f + "_log1p"] = {"value": amount, "available_at": end_of_day(d)}
        features[f + "_present"] = {"value": int(value is not None), "available_at": end_of_day(d)}
        previous = timeline.get(d - timedelta(days=1))
        prev_val = previous["smart"][f] if previous else None
        can_change = value is not None and prev_val is not None
        delta = math.log1p(value) - math.log1p(prev_val) if can_change else 0.
        features[f + "_delta_1d"] = {"value": delta, "available_at": end_of_day(d)}
        features[f + "_delta_present"] = {"value": int(can_change), "available_at": end_of_day(d)}
    return features


def future_status(timeline, d, horizon):
    """Positive only for an observed failure; negative only for complete future.
    A missing daily snapshot is censorship; disappearance is never a negative.
    """
    for step in range(1, horizon + 1):
        t = d + timedelta(days=step)
        following = timeline.get(t)
        if following is None:
            return "RIGHT_CENSORED", None, t
        if following["failed"]:
            return "POSITIVE", 1, t
    return "NEGATIVE", 0, d + timedelta(days=horizon)


def make_rows(manifest, timelines, phase):
    first = day(manifest["enrollment_date"])
    fit_end = day(manifest["fit_end"])
    dev_end = day(manifest["dev_end"])
    begin, end = (first, fit_end) if phase == "fit" else (fit_end, dev_end)
    labeled, case_index, censored = [], [], []
    for (serial, model), timeline in sorted(timelines.items()):
        ident = hashlib.sha256((model + "\0" + serial).encode()).hexdigest()
        for d in sorted(timeline):
            if not begin <= d < end or timeline[d]["failed"]:
                continue
            status, outcome, event_day = future_status(timeline, d, manifest["horizon_days"])
            descriptor = {"row_index": len(labeled), "entity_sha256": ident,
                          "model": model, "observation_day": d.isoformat(),
                          "outcome_day": event_day.isoformat(),
                          "label_status": status, "phase": phase,
                          "source_time_authentication": "NOT_INDEPENDENTLY_VERIFIED"}
            if status == "RIGHT_CENSORED":
                censored.append({k:v for k,v in descriptor.items() if k != "row_index"})
                continue
            row = {"observed_at": end_of_day(d),
                   "features": features_for(timeline, d, manifest["smart_fields"]),
                   "outcome": outcome, "outcome_at": end_of_day(event_day)}
            labeled.append(row)
            case_index.append(descriptor)
    return labeled, case_index, censored


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--csv-root", type=Path, required=True)
    p.add_argument("--phase", choices=["fit", "dev"], required=True)
    p.add_argument("--out-prefix", type=Path, required=True)
    args = p.parse_args(argv)
    try:
        files = [Path(str(args.out_prefix) + ext) for ext in
                 (".labeled.jsonl", ".index.jsonl", ".censored.jsonl", ".receipt.json")]
        if any(f.exists() for f in files):
            raise SourceError("output already exists, refusing to overwrite evidence")
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
        timelines, count = load_selected(manifest, args.phase, args.csv_root)
        labeled, indices, censored = make_rows(manifest, timelines, args.phase)
        args.out_prefix.parent.mkdir(parents=True, exist_ok=True)
        for path, rows in zip(files[:3], (labeled, indices, censored)):
            with path.open("x", encoding="utf-8") as output:
                for row in rows:
                    output.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
        receipt = {"status": "SOURCE_BYTES_VERIFIED_RIGHT_CENSORING_PROPOSED",
                   "qualification": "NO_TRAINED_MODEL_NO_INDEPENDENT_SOURCE_TIME_VALIDATION",
                   "phase": args.phase, "manifest_sha256": sha256(args.manifest),
                   "original_snapshot_rows_selected": count, "labeled": len(labeled),
                   "right_censored": len(censored), "observed_positives": sum(x["outcome"] for x in labeled),
                   "outputs": {p.name: sha256(p) for p in files[:3]}}
        with files[3].open("x", encoding="utf-8") as out:
            json.dump(receipt, out, indent=2, sort_keys=True)
        print(json.dumps(receipt, sort_keys=True))
        return 0
    except (OSError, SourceError, ValueError, KeyError, TypeError) as exc:
        print(json.dumps({"status": "BLOCKED_OR_FAILED", "reason": str(exc)}), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
