#!/usr/bin/env python3
"""Frozen 2014 Backblaze longitudinal cohort protocol - no source data opened.

The official 2014 archive is ~587 MB compressed. This module only registers
split/enrollment/censor/score contracts, tests fabricated fixtures, and refuses
any historic-source skill claim without future original byte/source receipts.
No actual drive data, labels, checkpoints, protected dates or GPU work herein.
"""
from __future__ import annotations
from datetime import date,timedelta
import hashlib
import json

SOURCE_URL="https://f001.backblazeb2.com/file/Backblaze-Hard-Drive-Data/data_2014.zip"
SOURCE_PUBLISHER="Backblaze Drive Stats"
SOURCE_LICENSE_NOTE="Publisher terms permit derivatives with Backblaze credit; do not resell raw data"
SOURCE_SHA256=None  # intentionally unset; DO NOT guess publisher source digest
HORIZON_DAYS=7
OBSERVATION_WARMUP_DAYS=7
FIT=(date(2014,1,1),date(2014,7,1))
CAL=(date(2014,7,1),date(2014,9,1))
DEV=(date(2014,9,1),date(2014,11,1))
PROTECTED_TEST=(date(2014,11,1),date(2015,1,1))
TIME_FIELDS=("publisher_snapshot_date","source_record_available_at","decision_asof")
COUNTERFACTUAL_LIMIT="no action logging or intervention outcome in source"
class ProtocolError(ValueError):pass

def phase(day):
    if FIT[0]<=day<FIT[1]:return "FIT"
    if CAL[0]<=day<CAL[1]:return "CAL"
    if DEV[0]<=day<DEV[1]:return "EXPOSED_DEV"
    if PROTECTED_TEST[0]<=day<PROTECTED_TEST[1]:return "PROTECTED_TEST"
    raise ProtocolError("outside frozen 2014 source period")

def admitted(history,day):
    """Admission only uses 7 observed prior/current nonfailure snapshots."""
    for n in range(OBSERVATION_WARMUP_DAYS):
        d=day-timedelta(days=n)
        if d not in history or history[d] is not False:
            return False
    return True

def fate(history,day):
    """Outcome only when all necessary successive days physically observed.

    An event day (flag=True) terminates follow-up. A missing preceding day
    makes outcome unknown/censored rather than labeling the case negative.
    Cross-split follow-up is forbidden by definition.
    """
    part=phase(day)
    limit={"FIT":FIT[1],"CAL":CAL[1],"EXPOSED_DEV":DEV[1],
           "PROTECTED_TEST":PROTECTED_TEST[1]}[part]
    if not admitted(history,day):
        raise ProtocolError("case not source history eligible")
    for n in range(1,HORIZON_DAYS+1):
        d=day+timedelta(days=n)
        if d>=limit or d not in history:
            return "RIGHT_CENSORED"
        if history[d] is True:
            return "RECORDED_FAILURE"
        if history[d] is not False:
            raise ProtocolError("publisher flag not boolean")
    return "OBSERVED_NO_FAILURE"

def training_allowed(manifest):
    """Fail-closed: this registration can never grant real training clearance."""
    if not isinstance(manifest,dict):
        return False
    return False

def frozen_registration():
    return {
        "schema":"model02-backblaze-2014-registered-protocol-v1",
        "source_publisher":SOURCE_PUBLISHER,
        "original_publisher_archive":SOURCE_URL,
        "publisher_archive_sha256":SOURCE_SHA256,
        "original_publisher_source_acquired":False,
        "original_source_publication_asof_per_drive_proven":False,
        "source_terms_and_independent_outcome_approval":False,
        "source_records_previously_evaluated_by_this_study":False,
        "fit_half_open":["2014-01-01","2014-07-01"],
        "cal_half_open":["2014-07-01","2014-09-01"],
        "dev_half_open":["2014-09-01","2014-11-01"],
        "heldout_test_half_open":["2014-11-01","2015-01-01"],
        "decision_horizon_days":HORIZON_DAYS,
        "predecision_consecutive_clean_source_days":OBSERVATION_WARMUP_DAYS,
        "admission": "after 7 source-observed consecutive current/prior nonfailure days, never selecting by later outcomes",
        "future_label_rule":"terminal recorded failure within next seven complete in-phase source-observed days; otherwise censor on disappearance, partition boundary or missing data",
        "source_timestamp_contract":list(TIME_FIELDS),
        "known_publisher_source_snapshot_not_realtime_availability":True,
        "source_manifest_not_equal_to_publication_clock":True,
        "heldout_TEST_access_authorized":False,
        "planned_contenders":[
            "FIT-prevalence constant risk",
            "causal current-day SMART feature logistic",
            "causal seven-day history summary classical model",
            "censor-aware survival baseline before model superiority",
            "matched 95k-105k native TAC vs GRU/gated-delta only when independent source and compute/data/parameter rules clear"
        ],
        "original_checkpoint_optimizer_rng_git_replay_compulsory":True,
        "evidence_requirements":["publisher source/archive SHA and per-day hashes",
          "manufacturer/model/drive identities for all models, with equal available information",
          "unique physical fail events and source lifecycle censoring",
          "right-censoring and left-truncation sensitivity",
          "effect on calibration, event-level recall at 1% operator alert capacity, cost per useful alert",
          "per-manufacturer and per-model subgroup errors",
          "source-visible-at temporal receipts or honest historical-only qualification",
          "independent case-level TEST access after freeze and explicit authorization"],
        "scientific_status":"SOURCE_REGISTRATION_ONLY_NO_TRAINING_OR_EVALUATION",
        "live_inference_and_drive_replacement_authorized":False,
        "limitations":[
            "source was published historically after snapshots; never presume timestamp means cloud available at the time",
            "no independent 2014 publisher ZIP bytes acquired or checkpoint fitted yet",
            "older Backblaze source studies inform the design; not a new unbiased dataset selection hypothesis",
            "dataset alone does not record the cost or outcome of repair/replacement interventions",
            COUNTERFACTUAL_LIMIT]}
if __name__=="__main__":
    print(json.dumps(frozen_registration(),indent=2,sort_keys=True))
