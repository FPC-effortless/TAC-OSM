#!/usr/bin/env python3
"""Independent FIT-only PRSA raw->ME111 13-feature Java/Python parity.

Reads the exact original public archive via the original source-pinned
me100_sensor_data.load('fit') implementation. This never reads 2015 REPORT,
2016-2017 protected numeric cohorts, target labels, or trains models.
The authentic 2013 source rows are inputs to a preprocessing test ONLY;
neither accuracy nor source-authorized live observations are claimed.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import zipfile

SOURCE_ARCHIVE_SHA = "d1b9261c54132f04c374f762f1e5e512af19f95c95fd6bfa1e8ac7e927e3b0b8"
OWNER_IDS = (0, 5)
EVENTS_PER_OWNER = 128

HARNESS = r"""
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.time.LocalDateTime;
import java.util.List;
import org.tacosm.mobile.me111.Me111Contract;
import org.tacosm.mobile.me111.Me111SourceFeatures;

public final class OriginalFitSourceHarness {
    static double maxAbs = 0.0;
    static int compared = 0;
    static void check(boolean condition, String message) {
        if (!condition) throw new AssertionError(message);
    }
    static double[] doubles(String cells, int count) {
        String[] parts = cells.split(",", -1);
        check(parts.length == count, "expected " + count + " reference fields");
        double[] v = new double[count];
        for (int i = 0; i < count; i++) v[i] = Double.parseDouble(parts[i]);
        return v;
    }
    static void compare(String cells, float[] actual, String name, int event) {
        double[] expected = doubles(cells, actual.length);
        for (int i = 0; i < actual.length; i++) {
            double error = Math.abs(expected[i] - actual[i]);
            if (error > maxAbs) maxAbs = error;
            check(Double.isFinite(actual[i]) && error <= 2e-5,
                 "frozen original " + name + " mismatch event=" + event
                 + " channel=" + i + " original=" + expected[i]
                 + " Java=" + actual[i]);
        }
    }
    static void reject(Runnable action) {
        try { action.run(); }
        catch (IllegalArgumentException correct) { return; }
        throw new AssertionError("invalid original source row was accepted");
    }
    public static void main(String[] args) throws Exception {
        List<String> lines = Files.readAllLines(Paths.get(args[0]), StandardCharsets.UTF_8);
        check(lines.size() == 3 + 2 * 128, "source parity fixture dimension");
        check(lines.get(0).equals("ME111_ORIGINAL_FIT_ONLY_RAW_PARITY_V1"),
              "original-source fixture format");
        Me111SourceFeatures encoder = new Me111SourceFeatures(
            doubles(lines.get(1), 13), doubles(lines.get(2), 13));
        Me111SourceFeatures.History history = null;
        int priorOwner = -1;
        for (int row = 3; row < lines.size(); row++) {
            String[] parts = lines.get(row).split("\\t", -1);
            check(parts.length == 9, "source fixture schema at " + row);
            int owner = Me111SourceFeatures.stationId(parts[0]);
            if (owner != priorOwner) {
                history = Me111SourceFeatures.beginningAtOriginalArchiveStart(owner);
                priorOwner = owner;
            }
            Me111Contract.Event event = encoder.encode(history, parts[0],
                LocalDateTime.parse(parts[1]), parts[2].split(",", -1), parts[3]);
            check(event.owner == owner && event.hour == Long.parseLong(parts[8]),
                  "source owner and exact epoch hour");
            compare(parts[4], event.values, "standardized values", row);
            String[] obs = parts[5].split(",", -1);
            check(obs.length == 13, "source masks");
            for (int i = 0; i < 13; i++)
                check(event.observed[i] == obs[i].equals("1"), "mask " + row + ":" + i);
            compare(parts[6], event.ages, "source ages", row);
            compare(parts[7], event.clock, "calendar encoding", row);
            compared++;
        }
        check(compared == 256, "actual source events compared");
        Me111SourceFeatures.History partial =
            Me111SourceFeatures.beginningAtOriginalArchiveStart(0);
        String[] missing = new String[11];
        java.util.Arrays.fill(missing, "NA");
        reject(() -> encoder.encode(partial, "Aotizhongxin",
            LocalDateTime.of(2013,3,1,1,0), missing, "NA"));
        check(partial.lastHour() == -1, "reject partial replay before changing state");
        reject(() -> encoder.encode(partial, "Changping",
            LocalDateTime.of(2013,3,1,0,0), missing, "NA"));
        reject(() -> encoder.encode(partial, "Aotizhongxin",
            LocalDateTime.of(2013,3,1,0,0), missing, "bad wind"));
        missing[0] = "-1";
        reject(() -> encoder.encode(partial, "Aotizhongxin",
            LocalDateTime.of(2013,3,1,0,0), missing, "NA"));
        check(partial.lastHour() == -1, "negative controls cannot advance age history");
        System.out.println("ME111_AUTHENTIC_FIT_FEATURE_PARITY_PASS=" + compared);
        System.out.println("ME111_AUTHENTIC_FIT_MAX_ABS_FEATURE_ERROR=" + maxAbs);
        System.out.println("ME111_AUTHENTIC_FIT_SCIENTIFIC_PREDICTION=NOT_EVALUATED");
    }
}
"""


def verify(source_root: Path) -> dict:
    import numpy as np
    script_source = source_root / "scripts/me100_sensor_data.py"
    archive = source_root / "research/source-bundles/ME100/PRSA2017_original.zip"
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    if digest != SOURCE_ARCHIVE_SHA:
        raise RuntimeError("original publisher archive hash mismatch")
    sys.path.insert(0, str(source_root / "scripts"))
    import me100_sensor_data as original
    if original.SHA != SOURCE_ARCHIVE_SHA or original.FIT_STOP <= 128:
        raise RuntimeError("source loader/fit cohort identity mismatch")
    d = original.load("fit")
    if d["phase"] != "fit":
        raise RuntimeError("source processing crossed into REPORT/TEST")
    root = Path(__file__).resolve().parents[1]
    java = (root / "mobile/android/model01/sample-app/app/src/main/java"
            / "org/tacosm/mobile/me111")
    means = d["mean"].astype(np.float64)
    std = d["std"].astype(np.float64)
    if means.shape != (13,) or std.shape != (13,) or not np.isfinite(means).all() or np.any(std < 0.0001):
        raise RuntimeError("unexpected pinned FIT moments")
    rows = ["ME111_ORIGINAL_FIT_ONLY_RAW_PARITY_V1",
            ",".join(repr(float(x)) for x in means),
            ",".join(repr(float(x)) for x in std)]
    attribution = "Chen, S. (2017), Beijing Multi-Site Air Quality, UCI, CC BY 4.0"
    with zipfile.ZipFile(archive) as original_zip:
        for owner in OWNER_IDS:
            site = original.SITES[owner]
            matches = [p for p in original_zip.namelist()
                       if p.endswith(f"PRSA_Data_{site}_20130301-20170228.csv")]
            if len(matches) != 1:
                raise RuntimeError("source station file identity ambiguous")
            stream = csv.DictReader(io.TextIOWrapper(
                original_zip.open(matches[0]), encoding="utf-8-sig"))
            for step, row in enumerate(stream):
                if step >= EVENTS_PER_OWNER:
                    break  # never process REPORT or protected years
                if row["station"] != site:
                    raise RuntimeError("source station name changed")
                assert all(int(row[k]) in range(1, 32) for k in ["day"])
                timestamp = (
                    f"{int(row['year']):04d}-{int(row['month']):02d}-"
                    f"{int(row['day']):02d}T{int(row['hour']):02d}:00:00"
                )
                input_numeric = ",".join(row[k] for k in original.NUMERIC)
                raw_wind = row["wd"]
                vals = d["values"][owner, step]
                obs = d["observed"][owner, step]
                ages = d["ages"][owner, step]
                clocks = d["clock"][owner, step]
                hour = int(d["hours"][owner, step])
                if hour != step or hour >= original.FIT_STOP:
                    raise RuntimeError("source chronology/fit partition changed")
                rows.append("\t".join([
                    site, timestamp, input_numeric, raw_wind,
                    ",".join(repr(float(x)) for x in vals),
                    ",".join("1" if x else "0" for x in obs),
                    ",".join(repr(float(x)) for x in ages),
                    ",".join(repr(float(x)) for x in clocks),
                    str(hour)
                ]))
    with tempfile.TemporaryDirectory(prefix="me111-source-parity-") as temporary:
        work = Path(temporary)
        records = work / "frozen-fit-source-records.tsv"
        records.write_text("\n".join(rows) + "\n", encoding="utf-8")
        harness = work / "OriginalFitSourceHarness.java"
        harness.write_text(HARNESS, encoding="utf-8")
        cp = subprocess.run([
            "javac", "-d", str(work), str(java / "Me111Contract.java"),
            str(java / "Me111SourceFeatures.java"), str(harness)
        ], check=True, capture_output=True, text=True, timeout=70)
        result = subprocess.run([
            "java", "-cp", str(work), "OriginalFitSourceHarness", str(records)
        ], check=True, capture_output=True, text=True, timeout=40)
    lines = result.stdout.splitlines()
    if "ME111_AUTHENTIC_FIT_FEATURE_PARITY_PASS=256" not in lines:
        raise RuntimeError("failed actual Java-vs-pinned-source parity")
    max_line = next(x for x in lines
                    if x.startswith("ME111_AUTHENTIC_FIT_MAX_ABS_FEATURE_ERROR="))
    max_error = float(max_line.split("=")[1])
    if max_error > 2e-5:
        raise RuntimeError("frozen source transformation numerical drift")
    return {
        "status": "PASS_ORIGINAL_FIT_SOURCE_256_RAW_ROWS_JAVA_PYTHON_PARITY",
        "original_source_sha256": SOURCE_ARCHIVE_SHA,
        "source_loader_sha256": hashlib.sha256(script_source.read_bytes()).hexdigest(),
        "source_attribution": attribution,
        "trained_checkpoint_changed": False,
        "training_performed": False,
        "source_rows_compared": 256,
        "stations": [original.SITES[k] for k in OWNER_IDS],
        "original_cohort": "FIT 2013-03-01 onward only",
        "report_test_access": "NONE",
        "original_fit_means": [float(v) for v in means],
        "original_fit_scales": [float(v) for v in std],
        "max_abs_feature_error": max_error,
        "new_scientific_forecast_claim": "NONE",
        "original_me111_scientific_gate": "FAILED_MIXED_TRADEOFF",
        "on_phone_raw_source_ingestion": "NOT_IMPLEMENTED",
        "physical_android": "NOT_RUN"
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-root", type=Path, required=True)
    p.add_argument("--receipt", type=Path, required=True)
    args = p.parse_args()
    if args.receipt.exists():
        p.error("refuse overwrite of authentic source preprocessing parity evidence")
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    try:
        receipt = verify(args.source_root.resolve())
    except Exception as error:
        receipt = {
            "status": "FAIL_ORIGINAL_FIT_SOURCE_RAW_PARITY",
            "reason_type": type(error).__name__,
            "reason": str(error)[:3000],
            "original_source_sha256": SOURCE_ARCHIVE_SHA,
            "new_scientific_forecast_claim": "NONE"
        }
    args.receipt.write_text(json.dumps(receipt, indent=2, sort_keys=True,
                                       allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(receipt, sort_keys=True))
    return 0 if receipt["status"].startswith("PASS_") else 2


if __name__ == "__main__":
    raise SystemExit(main())
