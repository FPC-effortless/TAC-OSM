#!/usr/bin/env python3
"""Reproducibly export the exact existing Model01 frozen ridge to offline Java.

No fitting, downloading, label access, model selection or device qualification.
The generated Java has no network, JSON, Android API or inference dependency.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "research/replay-bundles/PRODUCT-MODEL01-UCI374-CLASSICAL"
DEFAULT_OUTPUT = ROOT / "mobile/android/model01/Model01OfflinePredictor.java"
MODEL_SHA = "cb756a734b026602bb13f6af11b08711e2029f78aad1e95bb469d55131bc6c82"
SOURCE_SHA = "2fccf354445d886e7917620b0195db1f3e3e34d5a067a93b844694a4c561255a"


def pinned_model():
    raw = (BUNDLE / "ORIGINAL_CHECKPOINT.json").read_bytes()
    if hashlib.sha256(raw).hexdigest() != MODEL_SHA:
        raise ValueError("original checkpoint bytes are missing or modified")
    model = json.loads(raw)
    result = json.loads((BUNDLE / "ORIGINAL_DEV_RESULTS.json").read_bytes())
    if (model.get("model_schema") != "model01-uci374-lag-ridge-original-v1"
        or model.get("source_zip_sha256") != SOURCE_SHA
        or model.get("features") != ["latest", "lag_1", "lag_6", "lag_144", "sin_day", "cos_day"]
        or model.get("horizon_steps") != 6
        or model.get("fit_examples") != 9717
        or result.get("source_zip_sha256") != SOURCE_SHA
        or result.get("scientific_promotion") != "NOT_QUALIFIED"):
        raise ValueError("original scientific provenance/schema does not match")
    groups = [model.get("coefficients"), model.get("normalizer_fit_means"), model.get("normalizer_fit_scales")]
    if any(not isinstance(g, list) for g in groups) or [len(g) for g in groups] != [7, 6, 6]:
        raise ValueError("invalid coefficient dimensions")
    if any(type(x) not in (int, float) or not math.isfinite(x) for g in groups for x in g):
        raise ValueError("invalid finite numerical coefficients")
    if any(x <= 0 for x in groups[2]):
        raise ValueError("invalid normalization scale")
    q = result.get("calibration_q_90_wh")
    if type(q) not in (int, float) or not math.isfinite(q) or q < 0:
        raise ValueError("invalid original calibration value")
    return model, float(q)


def literal(values):
    return ", ".join(repr(float(v)) for v in values)


def render(model, q):
    """Return deterministic UTF-8 Java source from original frozen coefficients."""
    source = JAVA_TEMPLATE
    replacements = {
        "@COEFFICIENTS@": literal(model["coefficients"]),
        "@MEANS@": literal(model["normalizer_fit_means"]),
        "@SCALES@": literal(model["normalizer_fit_scales"]),
        "@Q@": repr(float(q)),
        "@MODEL_SHA@": MODEL_SHA,
        "@SOURCE_SHA@": SOURCE_SHA,
    }
    for name, value in replacements.items():
        source = source.replace(name, value)
    return source


JAVA_TEMPLATE = '''// Generated from the pinned original real-source ridge checkpoint. DO NOT HAND EDIT.
// Generate/check: python scripts/export_model01_android_java.py [--check]
// Classical model, NOT TAC-OSM neural; NOT independent scientific/device qualification.
package org.tacosm.mobile.model01;

/** No dependencies, I/O, sockets, cloud processing, embeddings or hidden fallback. */
public final class Model01OfflinePredictor {
    public static final String CHECKPOINT_SHA256 = "@MODEL_SHA@";
    public static final String SOURCE_SHA256 = "@SOURCE_SHA@";
    public static final String INDEPENDENT_QUALIFICATION = "NOT_QUALIFIED";
    public static final String ACTION_AUTHORITY = "ADVISORY_ONLY";
    public static final String INTERVAL_STATUS = "EMPIRICAL_HISTORICAL_CAL_NOT_GUARANTEED";
    public static final int HORIZON_MINUTES = 60;
    public static final int HISTORY_LENGTH = 145;
    private static final long MAX_SOURCE_INDEX = 20_000_000L;
    private static final double[] WEIGHTS = {@COEFFICIENTS@};
    private static final double[] MEANS = {@MEANS@};
    private static final double[] SCALES = {@SCALES@};
    private static final double CAL_Q_90_WH = @Q@;

    private Model01OfflinePredictor() { }

    private static void requireReading(double reading) {
        if (Double.isNaN(reading) || Double.isInfinite(reading)
                || reading < 0.0 || reading > 1_000_000.0) {
            throw new IllegalArgumentException("invalid observed energy Wh");
        }
    }

    private static void requireIndex(long index) {
        if (index < 144 || index > MAX_SOURCE_INDEX) {
            throw new IllegalArgumentException("unverified source-aligned index");
        }
    }

    /** Caller must establish the original dataset's ten-minute phase origin;
     * arbitrary device wall-clock timestamps are NOT interchangeable. */
    public static Prediction predict(long sourceAlignedIndex, double[] last145EnergyWh) {
        requireIndex(sourceAlignedIndex);
        if (last145EnergyWh == null || last145EnergyWh.length != HISTORY_LENGTH) {
            throw new IllegalArgumentException("exactly 145 historical readings are required");
        }
        for (double reading : last145EnergyWh) {
            requireReading(reading);
        }
        final double angle = 2.0 * Math.PI * (sourceAlignedIndex % 144L) / 144.0;
        final double[] features = {
            last145EnergyWh[144], last145EnergyWh[143], last145EnergyWh[138],
            last145EnergyWh[0], Math.sin(angle), Math.cos(angle)
        };
        double sum = 0.0;
        for (int j = 0; j < 6; j++) {
            sum += WEIGHTS[j + 1] * (features[j] - MEANS[j]) / SCALES[j];
        }
        double point = Math.max(0.0, WEIGHTS[0] + sum);
        if (Double.isNaN(point) || Double.isInfinite(point)) {
            throw new IllegalArgumentException("nonfinite prediction; abstain locally");
        }
        return new Prediction(point, point - CAL_Q_90_WH, point + CAL_Q_90_WH);
    }

    public static final class Prediction {
        public final double predictionWh;
        public final double interval90LowerWh;
        public final double interval90UpperWh;
        public final int horizonMinutes = HORIZON_MINUTES;
        public final String independentQualification = INDEPENDENT_QUALIFICATION;
        public final String intervalStatus = INTERVAL_STATUS;
        public final String actionAuthority = ACTION_AUTHORITY;

        private Prediction(double point, double low, double high) {
            this.predictionWh = point;
            this.interval90LowerWh = low;
            this.interval90UpperWh = high;
        }
    }

    /** Local, bounded, source-aligned observation buffer (NOT TAC recurrent memory).
     * Persist index+snapshot in app-private storage for restart continuity.
     * Missing/late observations require abstention, not silent interpolation. */
    public static final class WindowState {
        private long sourceAlignedIndex;
        private final double[] readings;

        public WindowState(long sourceAlignedIndex, double[] last145EnergyWh) {
            // Full validation before retaining a defensive copy.
            Model01OfflinePredictor.predict(sourceAlignedIndex, last145EnergyWh);
            this.sourceAlignedIndex = sourceAlignedIndex;
            this.readings = last145EnergyWh.clone();
        }

        public long sourceAlignedIndex() { return sourceAlignedIndex; }
        public double[] snapshot() { return readings.clone(); }
        public Prediction predict() {
            return Model01OfflinePredictor.predict(sourceAlignedIndex, readings);
        }

        public void append(long nextSourceAlignedIndex, double observedWh) {
            if (sourceAlignedIndex >= MAX_SOURCE_INDEX
                    || nextSourceAlignedIndex != sourceAlignedIndex + 1) {
                throw new IllegalArgumentException("source index gap/duplicate: abstain");
            }
            requireReading(observedWh);
            System.arraycopy(readings, 1, readings, 0, HISTORY_LENGTH - 1);
            readings[HISTORY_LENGTH - 1] = observedWh;
            sourceAlignedIndex = nextSourceAlignedIndex;
        }
    }
}
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="fail if committed export differs")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    model, q = pinned_model()
    expected = render(model, q).encode("utf-8")
    if args.check:
        if not args.output.is_file() or args.output.read_bytes() != expected:
            parser.error("generated Android Java is missing or differs from pinned model")
        print("MODEL01_ANDROID_EXPORT=PASS (pinned original and byte-exact generated source)")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_bytes(expected)
        print("MODEL01_ANDROID_EXPORT=WRITTEN (no device test)")


if __name__ == "__main__":
    main()
