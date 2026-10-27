#!/usr/bin/env python3
"""Real UCI Appliances Energy source-locked, DEV-ONLY baseline.

No TAC-OSM training, no opened TEST rows, no independent model qualification.
Source timestamps are publisher's naive local strings, NOT verified UTC or
real-time publication availability. Do not claim deployed real-time performance.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timedelta
import hashlib
import io
import json
import math
from pathlib import Path, PurePosixPath
import sys
import urllib.request
import zipfile

URL = "https://archive.ics.uci.edu/static/public/374/appliances+energy+prediction.zip"
DOI = "10.24432/C5VC8G"
ROWS = 19735
FIT_END = 9867  # first 50%; reserved for possible future FIT, unused here
CAL_END = 13814  # 70% (exposed CAL)
DEV_END = 15788  # 80% (exposed DEV); last 3947 rows TEST never parsed
H = 6  # 6 x 10-minute observations = one-hour-ahead
SEASON_LAG = 144  # previous day, also observed at decision time
MAX_ZIP_BYTES = 25_000_000
MAX_CSV_BYTES = 30_000_000


class SourceError(ValueError):
    pass


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def fetch_official_zip():
    req = urllib.request.Request(URL, headers={"User-Agent": "TAC-OSM-source-audit/1.0"})
    with urllib.request.urlopen(req, timeout=25) as response:
        if response.geturl() != URL:
            raise SourceError("unexpected redirect outside pinned original source URL")
        buf = bytearray()
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            buf.extend(chunk)
            if len(buf) > MAX_ZIP_BYTES:
                raise SourceError("download exceeds fixed byte budget")
    return bytes(buf)


def original_rows(zip_bytes: bytes):
    if len(zip_bytes) > MAX_ZIP_BYTES:
        raise SourceError("oversized publisher archive")
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as archive:
        choices = []
        for info in archive.infolist():
            path = PurePosixPath(info.filename)
            if info.filename.startswith("/") or ".." in path.parts or info.is_dir():
                continue
            if path.name == "energydata_complete.csv":
                if info.file_size > MAX_CSV_BYTES:
                    raise SourceError("publisher CSV exceeds uncompressed budget")
                choices.append(info)
        if len(choices) != 1:
            raise SourceError("expected exactly one original energydata_complete.csv")
        with archive.open(choices[0], "r") as raw:
            reader = csv.DictReader(io.TextIOWrapper(raw, encoding="utf-8-sig", newline=""))
            fields = set(reader.fieldnames or [])
            if not {"date", "Appliances"}.issubset(fields):
                raise SourceError("unexpected original dataset schema")
            # Use exactly the first 80% (including FIT/CAL/DEV). The untouched
            # future TEST tail is not parsed, scored, printed or inspected.
            prior = None
            out = []
            for idx, row in enumerate(reader):
                if idx == DEV_END:
                    break
                if row is None or not row.get("date"):
                    raise SourceError("missing publisher timestamp")
                try:
                    when = datetime.strptime(row["date"], "%Y-%m-%d %H:%M:%S")
                    y = float(row["Appliances"])
                except (TypeError, ValueError) as exc:
                    raise SourceError("invalid publisher date or energy") from exc
                if not math.isfinite(y) or y < 0:
                    raise SourceError("invalid energy target")
                if prior is not None and when - prior != timedelta(minutes=10):
                    raise SourceError("non-contiguous original 10-minute series")
                out.append(y)
                prior = when
            if len(out) != DEV_END:
                raise SourceError("unexpected source sample count in exposure partition")
    return out


def candidate_forecast(values, index, method):
    if method == "last_observation":
        return values[index]
    if method == "same_time_previous_day":
        if index < SEASON_LAG:
            raise SourceError("not enough legitimate past data")
        return values[index - SEASON_LAG]
    raise SourceError("unsupported model")


def eligible_indices(start, end):
    if not (FIT_END <= start < end <= DEV_END):
        raise SourceError("access outside exposed CAL or DEV")
    return range(start, end - H)


def quantile(residuals, alpha=0.1):
    if not residuals:
        raise SourceError("no calibrated residuals")
    if not 0 < alpha < 1:
        raise SourceError("invalid alpha")
    n = len(residuals)
    rank = math.ceil((n + 1) * (1 - alpha))
    if rank > n:
        raise SourceError("finite-sample quantile rank unavailable")
    return sorted(residuals)[rank - 1], rank


def metrics(values, indices, method, correction):
    total_abs = 0.
    total_sq = 0.
    hit = 0
    interval_score = 0.
    count = 0
    alpha = .1
    for i in indices:
        target_index = i + H
        if target_index >= DEV_END:
            raise SourceError("protected TEST label access denied")
        p = candidate_forecast(values, i, method)
        actual = values[target_index]
        diff = abs(p - actual)
        total_abs += diff
        total_sq += diff * diff
        lo, hi = p - correction, p + correction
        hit += lo <= actual <= hi
        interval_score += (hi - lo) + 2 / alpha * max(lo - actual, actual - hi, 0)
        count += 1
    if not count:
        raise SourceError("no evaluation cases")
    return {"n": count, "mae_wh": total_abs / count,
            "rmse_wh": math.sqrt(total_sq / count),
            "interval_width_wh": 2 * correction, "coverage_90_observed": hit / count,
            "interval_score_90": interval_score / count}


def score_source(zip_bytes):
    values = original_rows(zip_bytes)
    cal = list(eligible_indices(FIT_END, CAL_END))
    dev = list(eligible_indices(CAL_END, DEV_END))
    methods = ["last_observation", "same_time_previous_day"]
    cal_mae = {}
    for method in methods:
        cal_mae[method] = sum(abs(candidate_forecast(values, i, method) - values[i + H])
                              for i in cal) / len(cal)
    selected = min(methods, key=lambda method: (cal_mae[method], method))
    residuals = [abs(candidate_forecast(values, i, selected) - values[i + H])
                 for i in cal]
    q, rank = quantile(residuals)
    measured = metrics(values, dev, selected, q)
    other = {}
    for method in methods:
        q_other, _ = quantile([abs(candidate_forecast(values, i, method) - values[i + H])
                               for i in cal])
        other[method] = metrics(values, dev, method, q_other)
    return {
        "study": "MODEL01-UCI374-EXPOSED-DEV-PREQUALIFICATION",
        "evidence_tier": "AUTHENTIC_SOURCE_DEV_ONLY_NO_INDEPENDENT_CONFIRMATION",
        "dataset": "UCI Appliances Energy Prediction",
        "original_source_url": URL,
        "citation_doi": DOI, "publisher_license": "CC BY 4.0; attribution required",
        "source_zip_sha256": digest(zip_bytes), "zip_size_bytes": len(zip_bytes),
        "source_time": "naive original local datetime, NOT verified availability clock",
        "original_case_count_publisher": ROWS,
        "fit_boundary_index_unused": FIT_END, "calibration_end_index": CAL_END,
        "exposed_dev_end_index": DEV_END, "protected_test_start_index": DEV_END,
        "protected_test_targets_parsed": False, "horizon_steps": H,
        "calibration_cases": len(cal), "dev_cases": len(dev),
        "selected_using_cal_only": selected, "calibration_mae_wh": cal_mae,
        "calibration_q_90_wh": q, "calibration_quantile_rank": rank,
        "selected_dev": measured, "all_candidate_dev": other,
        "trained_tac_model": False, "independent_test_scored": False,
        "scientific_promotion": "NOT_QUALIFIED",
        "disclosures": [
            "historical source release does not certify original decision-time data availability",
            "single-household and overlapping forecast windows violate independent-case assumptions",
            "time-series dependence can invalidate nominal finite-sample exchangeability coverage",
            "DEV results are exposed after this run and cannot be retuned as independent confirmation",
            "past known public data, not a prospective real-time deployment, device or cloud benchmark"
        ],
    }


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-file", type=Path,
                   help="optional authentic publisher ZIP already acquired by operator")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args(argv)
    try:
        if args.out.exists():
            raise SourceError("refusing overwrite of score evidence")
        data = args.source_file.read_bytes() if args.source_file else fetch_official_zip()
        report = score_source(data)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("x", encoding="utf-8") as f:
            json.dump(report, f, indent=2, sort_keys=True, allow_nan=False)
        print(json.dumps(report, sort_keys=True))
        return 0
    except (SourceError, ValueError, OSError, UnicodeError, zipfile.BadZipFile,
            TimeoutError) as exc:
        print(json.dumps({"status": "BLOCKED_OR_FAILED", "reason": str(exc)}), file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
