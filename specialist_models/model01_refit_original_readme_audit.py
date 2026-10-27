#!/usr/bin/env python3
"""Audit publisher REFIT original README for native sample-provenance limitations.

Publisher original 2015 REFIT data records change-triggered, polled appliance
power; intermediate 8s values are filled. 'Raw' source CSV must NOT be called
75 independent actually observed IAM readings without source-native hardware
provenance. No large dataset is downloaded or training/scoring performed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
import urllib.parse
import urllib.request

URL="https://pure.strath.ac.uk/ws/portalfiles/portal/45410335/REFITREADME.txt"
LANDING="https://pureportal.strath.ac.uk/en/datasets/refit-electrical-load-measurements/"
DOI="10.15129/31da3ece-f902-4e95-a093-e0a9536983c4"
MAX_SIZE=48_000
ESSENTIAL={
    "source_change_triggered":"data was only recorded when there was a change in load",
    "intermediate_values_filled":"this data has been filled with intermediate values where not available",
    "polling_not_synchronized":"the sensors are also not synchronised",
    "sensor_timestamp_lag":"sensor may have updated anywhere in the last 6 seconds",
    "appliance_may_be_switched":"we cannot guarantee this to be the case"
}

class ProvenanceError(ValueError):
    pass

def fetch_publisher_readme():
    req=urllib.request.Request(URL,headers={"User-Agent":"TAC-OSM-original-REFIT-source-attestation/1"})
    with urllib.request.urlopen(req,timeout=25) as reply:
        target=urllib.parse.urlparse(reply.geturl())
        if target.scheme!="https" or target.hostname not in ("pureportal.strath.ac.uk","pure.strath.ac.uk"):
            raise ProvenanceError("unexpected README redirect origin")
        data=reply.read(MAX_SIZE+1)
        if len(data)>MAX_SIZE:
            raise ProvenanceError("publisher README exceeds bounded budget")
    return data

def inspect(data):
    if not isinstance(data,bytes) or not 500<len(data)<=MAX_SIZE:
        raise ProvenanceError("invalid original publisher README bytes")
    try:
        original=data.decode("utf-8-sig")
    except UnicodeError as e:
        raise ProvenanceError("invalid source UTF-8") from e
    plain=re.sub(r"\s+"," ",original.casefold())
    matches={k:phrase in plain for k,phrase in ESSENTIAL.items()}
    if not all(matches.values()) or "house 1" not in plain or "licenc" not in plain:
        raise ProvenanceError("native source provenance language changed")
    return {
        "schema":"model01-refit-original-publisher-readme-observability-v1",
        "original_unmodified_README_sha256":hashlib.sha256(data).hexdigest(),
        "original_source_readme_url":URL,
        "source_landing_page":LANDING,
        "source_raw_dataset_doi":DOI,
        "publisher":"University of Strathclyde",
        "original_source_license":"CC BY 4.0 stated in original README",
        "verified_publisher_text_flags":matches,
        "eight_second_interval_represents_independent_observation":False,
        "native_8second_actually_observed_mask_in_plain_CSV":False,
        "source_available_at_the_time_proven":False,
        "per_appliance_identity_unchanged_proven":False,
        "appliance_target_equivalence_to_UCI374":False,
        "original_raw_archive_downloaded":False,
        "model_fitted_or_evaluated":False,
        "scientific_qualification":"BLOCKED",
        "next":"find truly sampled native publisher time records or original signal-change events/observability flags, reconcile IAM appliance movements, and freeze another independently observable outcome target before evaluating"
    }

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args(argv)
    try:
        if a.out.exists():
            raise ProvenanceError("refuse to overwrite original source receipt")
        data=fetch_publisher_readme()
        result=inspect(data)
        a.out.parent.mkdir(parents=True,exist_ok=True)
        a.out.write_text(json.dumps(result,sort_keys=True,indent=2)+"\n")
        print(json.dumps(result,sort_keys=True))
        return 0
    except (OSError,ProvenanceError,ValueError,TypeError) as error:
        print(json.dumps({"status":"SOURCE_AUDIT_BLOCKED","reason":str(error)}),file=sys.stderr)
        return 2

if __name__=="__main__":
    sys.exit(main())
