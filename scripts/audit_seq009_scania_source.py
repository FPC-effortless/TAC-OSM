#!/usr/bin/env python3
"""SEQ009 real SCANIA v3 source acquisition/audit: NO model fitting or TEST access.

Public publisher source files ONLY. Metadata/counts/hashes are evidence that
authentic bytes were acquired, not evidence of predictive model capability.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import time
import urllib.error
import urllib.parse
import urllib.request

DOI = "10.5878/bnh5-ka77"
DATASET_URL = "https://researchdata.se/en/catalogue/dataset/2024-34/3"
PUBLISHER_BASE = "https://api.researchdata.se/dataset/2024-34/3/file/data"
FILES = ("train_tte.csv", "train_specifications.csv", "validation_labels.csv")
EXPECTED = {
    "train_tte.csv": {"vehicle_id", "length_of_study_time_step", "in_study_repair"},
    "train_specifications.csv": {"vehicle_id"},
    "validation_labels.csv": {"vehicle_id", "class_label"},
}
# Deliberately excludes test_labels, test history, and any full-repair labels.
LIMIT_BYTES = {"train_tte.csv": 3_000_000,
               "train_specifications.csv": 4_000_000,
               "validation_labels.csv": 1_000_000}

# Archived publisher ZIP route was GET-verified in CI; individual file endpoint
# returned HTTP 404 and is deliberately not used for scientific provenance.
from acquire_seq009_scania_zip import acquire


def audit_csv(path, name):
    if name not in FILES:
        raise PermissionError("Only approved TRAIN/VALIDATION files")
    raw = path.read_bytes()
    text = raw.decode("utf-8-sig")
    reader = csv.DictReader(io.StringIO(text))
    columns = set(reader.fieldnames or ())
    missing = EXPECTED[name] - columns
    if missing:
        raise ValueError(f"Schema mismatch for {name}: missing {sorted(missing)}; got {sorted(columns)}")
    ids, positive, labels = set(), 0, {}
    n = 0
    for row in reader:
        n += 1
        vehicle_id = (row.get("vehicle_id") or "").strip()
        if not vehicle_id:
            raise ValueError(f"{name} row {n}: empty vehicle_id")
        if vehicle_id in ids:
            raise ValueError(f"{name}: duplicate vehicle_id {vehicle_id}")
        ids.add(vehicle_id)
        if name == "train_tte.csv":
            flag = (row.get("in_study_repair") or "").strip()
            if flag not in ("0", "1"):
                raise ValueError("Unrecognized repair flag")
            positive += (flag == "1")
            try:
                dur = float(row["length_of_study_time_step"])
            except (ValueError, TypeError) as exc:
                raise ValueError("Invalid event/censor duration") from exc
            if not (dur >= 0 and dur < float("inf")):
                raise ValueError("Invalid duration")
        if name == "validation_labels.csv":
            label = str(row.get("class_label", "")).strip()
            if label not in {"0", "1", "2", "3", "4"}:
                raise ValueError(f"Unexpected class_label {label}")
            labels[label] = labels.get(label, 0) + 1
    if n == 0:
        raise ValueError(f"{name} contains no data")
    report = {
        "name": name, "rows": n, "unique_vehicle_ids": len(ids),
        "columns": sorted(columns),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }
    if name == "train_tte.csv":
        report.update(repaired=positive, censored=n-positive,
                      empirical_repair_fraction=positive / n)
    if name == "validation_labels.csv":
        report["class_counts"] = labels
    return report, ids


def audit(root, manifest):
    if manifest["dataset_doi"] != DOI or manifest["version"] != 3:
        raise ValueError("Version/DOI mismatch")
    reports, ids = {}, {}
    for name in FILES:
        path = root / name
        if not path.is_file():
            raise FileNotFoundError(f"Verified real publisher file REQUIRED: {name}")
        report, unique_ids = audit_csv(path, name)
        if report["sha256"] != manifest["files"][name]["sha256"]:
            raise ValueError(f"File altered since original publisher acquisition: {name}")
        reports[name], ids[name] = report, unique_ids
    if ids["train_tte.csv"] != ids["train_specifications.csv"]:
        raise ValueError("TRAIN vehicle identity mismatch between TTE and specifications")
    if ids["train_tte.csv"] & ids["validation_labels.csv"]:
        raise ValueError("TRAIN/VALIDATION vehicle ID overlap")
    # Observed counts are printed rather than silently coerced to publication numbers.
    return {"dataset_doi": DOI, "dataset_version": 3,
            "source_type": "authentic_publisher_files",
            "models_fitted": 0, "test_labels_opened": False,
            "commercial_qualification": False, "files": reports}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", default="results/seq009-source")
    args = p.parse_args()
    output = Path(args.output)
    root = output / "raw"
    acquisition = {"dataset_doi": DOI, "version": 3,
                   "catalogue": DATASET_URL, "files": {},
                   "test_labels_opened": False}
    provenance = acquire(output)
    acquisition["files"] = provenance["extracted_files"]
    acquisition["archive"] = {k: v for k, v in provenance.items() if k != "extracted_files"}
    for name in FILES:
        print(f"publisher-zip member verified: {name}", flush=True)
    outcome = audit(root, acquisition)
    output.mkdir(parents=True, exist_ok=True)
    (output / "acquisition.json").write_text(
        json.dumps(acquisition, sort_keys=True, indent=2) + "\n")
    (output / "audit.json").write_text(
        json.dumps(outcome, sort_keys=True, indent=2) + "\n")
    print(json.dumps({"status": "REAL_SOURCE_AUDITED", "models_fitted": 0,
                      "test_labels_opened": False,
                      "files": list(outcome["files"]),
                      "train_repaired": outcome["files"]["train_tte.csv"]["repaired"]}),
          flush=True)


if __name__ == "__main__":
    main()
