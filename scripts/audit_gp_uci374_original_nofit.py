#!/usr/bin/env python3
"""GP-UNC-004 original UCI374 source/inference event-stream NO-FIT audit.

This replays authentic ALREADY-EXPOSED calibration inputs through an ORIGINAL
preexisting classical ridge checkpoint and the truth-isolated streaming GP
boundary with ADAPTATION DISABLED. It measures software wiring, NOT new
predictive skill, learned TAC-OSM performance or independent qualification.

The downloaded historical publisher archive is pre-parsed by its pinned source
reader; index-based 10-minute label arrivals are an *experimental convention*,
NOT independently verified original publication/operational availability.
The only GP output correction is its zero-mean prior (no residual conditioning).
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

from tac_osm.gp_event_boundary_nofit import replay_truth_isolated_constructed

ORIGINAL_SOURCE_COMMIT = "66cc1ffc1935bad67118962091a016e04c2eff81"
SOURCE_SHA256 = "2fccf354445d886e7917620b0195db1f3e3e34d5a067a93b844694a4c561255a"
CHECKPOINT_SHA256 = "cb756a734b026602bb13f6af11b08711e2029f78aad1e95bb469d55131bc6c82"
CASES = 64


def _safe_checkout(path: Path):
    path = path.resolve()
    if not path.is_dir() or not (path / "specialist_models" / "model01_reference_api.py").is_file():
        raise ValueError("pinned original source checkout missing")
    actual = subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()
    if actual != ORIGINAL_SOURCE_COMMIT:
        raise ValueError("unverified original classical model source version")
    return path


def audit(checkout: Path):
    checkout = _safe_checkout(checkout)
    sys.path.insert(0, str(checkout / "specialist_models"))
    import uci_appliances_real_dev as uci
    import model01_reference_api as api
    if (uci.FIT_END != 9867 or uci.CAL_END != 13814 or
            uci.DEV_END != 15788 or uci.H != 6 or
            api.ORIGINAL_MODEL_SHA256 != CHECKPOINT_SHA256 or
            api.ORIGINAL_SOURCE_SHA256 != SOURCE_SHA256):
        raise ValueError("unexpected source identity / partition / horizon")
    archive = uci.fetch_official_zip()
    actual_digest = hashlib.sha256(archive).hexdigest()
    if actual_digest != SOURCE_SHA256:
        raise ValueError("source bytes differ from pinned official UCI archive")
    values = uci.original_rows(archive)  # parses exposed FIT/CAL/DEV, no TEST
    model, original_calibration_q = api.load_original()
    if model.get("model_type") != "classical_fit_only_ridge":
        raise ValueError("original source model is not the frozen classical baseline")

    start, stop = uci.FIT_END, uci.FIT_END + CASES
    if stop + uci.H > uci.CAL_END or start < 144:
        raise ValueError("must remain strictly inside exposed CAL label range")
    issued = {}
    def events():
        for index in range(start, stop + uci.H):
            if index < stop:
                request = {
                    "schema": "model01-energy-source-aligned-v1",
                    "source_aligned_step_index": index,
                    "last_145_energy_wh": values[index-144:index+1],
                }
                prediction = api.predict(model, original_calibration_q, request)
                if (prediction["model_checkpoint_sha256"] != CHECKPOINT_SHA256 or
                        prediction["reference_source_sha256"] != SOURCE_SHA256):
                    raise ValueError("original frozen forecast identity mismatch")
                point = prediction["prediction_wh"]
                issued[index] = point
                yield {
                    "kind": "issue", "row_id": str(index),
                    "entity_id": "single-original-household",
                    "features": (float(index % 144)/144,),
                    "base_prediction": point, "feature_at": float(index),
                    "issued_at": float(index),
                    "checkpoint_sha256": CHECKPOINT_SHA256,
                }
            past = index - uci.H
            if past in issued:
                # Only a distinct later stream event introduces the archived label.
                yield {"kind": "label_arrived", "row_id": str(past),
                       "available_at": float(index), "truth": values[index]}
    records = replay_truth_isolated_constructed(
        events(), {"single-original-household": [[0.0], [0.5], [1.0]]},
        checkpoint_sha256=CHECKPOINT_SHA256,
        adaptive_feedback=False, max_pending=16, max_records=CASES)
    if len(records) != CASES or len(issued) != CASES:
        raise ValueError("original source did not produce the frozen case count")
    for row in records:
        index = int(row.row_id)
        if (row.base_prediction != issued[index] or
                abs(row.corrected_prediction - issued[index]) > 1e-12 or
                row.eligible_past_updates != 0):
            raise ValueError("source forecast parity or no-adaptation contract failure")
    return {
        "schema": "gp-unc-004.authentic-exposed-cal-nofit.v1",
        "status": "PASS_ORIGINAL_SOURCE_READ_ONLY_EVENT_WIRING",
        "source_commit": ORIGINAL_SOURCE_COMMIT,
        "source_zip_sha256": actual_digest,
        "classical_checkpoint_sha256": CHECKPOINT_SHA256,
        "source": "UCI Appliances Energy Prediction DOI 10.24432/C5VC8G",
        "source_model": "previously trained classical ridge, NOT neural TAC-OSM",
        "cohort": "ALREADY_EXPOSED_CAL_ONLY",
        "issue_indices": [start, stop],
        "label_indices": [start+uci.H, stop+uci.H],
        "forecast_events": CASES,
        "original_label_events": CASES,
        "gp_updated_on_real_labels": False,
        "gp_prior_mean_correction_zero": True,
        "all_original_api_predictions_preserved": True,
        "source_numeric_parsed_through_exposed_dev": True,
        "protected_test_parsed": False,
        "time_assumption": "original_10_minute_index_NOT_VERIFIED_REAL_AVAILABILITY",
        "new_fit": False,
        "new_source_qualification": False,
        "independent_test": False,
        "new_prediction_accuracy_claim": False,
        "scientific_promotion": "NOT_QUALIFIED",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pinned-checkout", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("refusing overwrite of original-source audit receipt")
    report = audit(args.pinned_checkout)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
    print(json.dumps(report, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
