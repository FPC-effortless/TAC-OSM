#!/usr/bin/env python3
"""UK-DALE publisher metadata audit and fail-closed source-meter admission.

Original publication: Kelly and Knottenbelt, Scientific Data 2015, DOI
10.1038/sdata.2015.7; final 2017 6s release DOI 10.5286/UKERC.EDC.000004.
CC BY 4.0 with attribution. Metadata pinned to original author Git commit.
Only publisher metadata is fetched online (five small YAML files); NO UK-DALE
power records or protected House 5 data are downloaded.

New task: 10-minute *sampled mean kettle plug active power (W)*, predicted
60 minutes ahead, trained/evaluated on house-disjoint real kettle channels.
NOT UCI374 'Appliances' Wh, not whole-home electricity, not reconstructed
meter Wh, not proof of source measurement/available-at sensor time.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import sys
import urllib.request

COMMIT="0d168958ea2e69b8bcb0dccc8a732adbfe085985"
BASE=f"https://raw.githubusercontent.com/JackKelly/UK-DALE_metadata/{COMMIT}/"
RELEASE_DOI="10.5286/UKERC.EDC.000004"
PAPER_DOI="10.1038/sdata.2015.7"
METERS={2:8,3:2,5:18}
ROLES={2:"FIT_AND_CHRONOLOGICAL_CAL",3:"DEV_NEW_HOUSE",5:"SEALED_TEST_METADATA_ONLY"}
EXCLUDED={1,4}
MIN_SAMPLES=90
MAX_GAP_SECONDS=15
MAX_METADATA_BYTES=125000

class AdmissionError(ValueError): pass

def _publisher_file(name):
    if name not in ["dataset.yaml"]+[f"building{j}.yaml" for j in range(1,6)]:
        raise AdmissionError("unlisted publisher metadata file")
    url=BASE+name
    req=urllib.request.Request(url,headers={"User-Agent":"TACOSM-UKDALE-source-audit/1"})
    with urllib.request.urlopen(req,timeout=20) as response:
        final=response.geturl()
        if final!=url:
            raise AdmissionError("publisher URI redirect not accepted")
        raw=response.read(MAX_METADATA_BYTES+1)
    if not 100<len(raw)<=MAX_METADATA_BYTES:
        raise AdmissionError("publisher source metadata missing / too large")
    return raw

def verify_metadata(dataset,buildings):
    if (not isinstance(dataset,dict) or dataset.get("name")!="UK-DALE"
        or dataset.get("number_of_buildings")!=5):
        raise AdmissionError("unexpected publisher dataset")
    license_list=dataset.get("rights_list",[])
    if not any("4.0" in str(x.get("name","")) and
               "Attribution" in str(x.get("name",""))
               for x in license_list if isinstance(x,dict)):
        raise AdmissionError("CC BY publisher rights metadata missing")
    if set(buildings)!={1,2,3,4,5}:
        raise AdmissionError("five complete author building records required")
    admitted={}
    rejected={}
    for house,meta in buildings.items():
        if not isinstance(meta,dict) or meta.get("instance")!=house:
            raise AdmissionError("publisher house mismatch")
        appliances=meta.get("appliances")
        meters=meta.get("elec_meters")
        if not isinstance(appliances,list) or not isinstance(meters,dict):
            raise AdmissionError("metadata lacks device/appliance identities")
        kettle=[p for p in appliances if isinstance(p,dict) and p.get("type")=="kettle"]
        if len(kettle)!=1 or not isinstance(kettle[0].get("meters"),list) or len(kettle[0]["meters"])!=1:
            raise AdmissionError("unknown or composite kettle instrument identity")
        channel=kettle[0]["meters"][0]
        co_metered=[a for a in appliances if channel in a.get("meters",[])]
        if house in EXCLUDED:
            if len(co_metered)<2:
                raise AdmissionError("previously shared kettle instrument became dedicated; re-freeze task")
            rejected[str(house)]={"reason":"KETTLE_SHARED_WITH_OTHER_DEVICES",
                                  "kettle_meter":channel,
                                  "co_metered_count":len(co_metered)}
            continue
        if channel!=METERS[house] or len(co_metered)!=1:
            raise AdmissionError("eligible kettle changed channel or shares its output")
        entry=meters.get(channel) or meters.get(str(channel))
        if (not isinstance(entry,dict) or entry.get("device_model")!="EcoManagerTxPlug"
            or entry.get("disabled") is True or entry.get("site_meter") is True
            or entry.get("data_location")!=f"house_{house}/channel_{channel}.dat"):
            raise AdmissionError("meter disabled/not a genuine individual power plug")
        clip=entry.get("preprocessing_applied",{}).get("clip",{}).get("upper_limit")
        if clip!=4000:
            raise AdmissionError("source meter clipping contract changed")
        admitted[str(house)]={
            "role":ROLES[house],"publisher_house":house,
            "meter_channel":channel,"original_file":entry["data_location"],
            "single_appliance_label":"kettle",
            "measures_active_power_watts":True,
            "publisher_values_preprocessed_clipped_at_watts":4000,
            "original_sampled_data_loaded":False,
            "sample_observed_at_physical_sensor_proven":False,
            "real_time_sensor_available_at_proven":False}
    return {"schema":"ukdale-kettle-five-house-2017-metadata-admission-v1",
            "metadata_commit":COMMIT,"dataset_publisher":"UK Energy Research Centre Energy Data Centre",
            "original_paper_doi":PAPER_DOI,"2017_disaggregated_release_doi":RELEASE_DOI,
            "publisher_metadata_license":"CC BY 4.0",
            "eligible_source_meters":admitted,"excluded_meters":rejected,
            "new_task_target":"10min logged-sample-mean kettle plug active power W",
            "original_UCI374_Appliances_target_equivalent":False,
            "original_UCI374_checkpoint_direct_transfer_claim":False,
            "holdout_house5_power_rows_opened":False,
            "source_archive_power_rows_downloaded":False,
            "source_log_time_is_physical_measurement_available_at_proven":False,
            "trained_model_evaluated":False,
            "source_qualification":"PUBLISHER_METADATA_VERIFIED_ONLY_NOT_POWER_RECORDS",
            "source_release_or_customer_ready":False}

def audit_publisher_metadata():
    try:
        import yaml
    except ImportError as exc:
        raise AdmissionError("PyYAML required for original author metadata") from exc
    raws={name:_publisher_file(name) for name in
          ["dataset.yaml"]+[f"building{k}.yaml" for k in range(1,6)]}
    contents={name:yaml.safe_load(data) for name,data in raws.items()}
    doc=verify_metadata(contents["dataset.yaml"],
                        {k:contents[f"building{k}.yaml"] for k in range(1,6)})
    doc["original_publisher_metadata_file_sha256"]={
        name:hashlib.sha256(contents).hexdigest() for name,contents in raws.items()}
    doc["original_publisher_metadata_urls"]={name:BASE+name for name in raws}
    return doc

def aggregate_kettle_snapshots(lines,house,manifest,source_sha256):
    """Original single channel, no imputation, gaps reject; never read house5."""
    if house not in (2,3):
        raise AdmissionError("test house must remain sealed until separate final registration")
    if manifest.get("metadata_commit")!=COMMIT or \
       manifest.get("schema")!="ukdale-kettle-five-house-2017-metadata-admission-v1":
        raise AdmissionError("publisher metadata source receipt mismatch")
    station=manifest.get("eligible_source_meters",{}).get(str(house))
    if not station or station.get("meter_channel")!=METERS[house]:
        raise AdmissionError("incorrect source house/meter pairing")
    if not re.fullmatch(r"[a-f0-9]{64}",source_sha256):
        raise AdmissionError("original extracted source SHA256 must be supplied")
    last=None
    current=None
    buf=[]
    results=[]
    digest=hashlib.sha256()
    n=0
    first=None
    last_seen=None
    for line in lines:
        if isinstance(line,str):
            raw=line.encode("ascii",errors="strict")
        elif isinstance(line,bytes):
            raw=line
        else:
            raise AdmissionError("unexpected original meter record type")
        digest.update(raw)
        n+=1
        if n>8_000_000:
            raise AdmissionError("bounded maximum kettle source lines exceeded")
        parts=raw.strip().split()
        if len(parts)!=2 or any(not v.isdigit() for v in parts):
            raise AdmissionError("original .dat source must contain UNIX timestamp and nonnegative watt integer")
        stamp=int(parts[0]);watts=int(parts[1])
        if stamp<1262304000 or stamp>1893456000 or not 0<=watts<=4000:
            raise AdmissionError("out-of-domain timestamp/publisher clip range")
        if last is not None and stamp<=last:
            raise AdmissionError("duplicate/backward UK-DALE source timestamp")
        slot=stamp//600
        if current is not None and slot!=current:
            calculated=complete_slot(current,buf)
            if calculated is not None:
                results.append(calculated)
            buf=[]
        buf.append((stamp,watts))
        current=slot
        last=stamp
        if first is None:first=stamp
        last_seen=stamp
    if current is not None:
        calculated=complete_slot(current,buf)
        if calculated is not None:results.append(calculated)
    if not n or digest.hexdigest()!=source_sha256:
        raise AdmissionError("raw kettle stream content hash missing/mismatched")
    if not results:raise AdmissionError("no fully observed 10-minute source log windows")
    if len(results)>350000:
        raise AdmissionError("unbounded source windows")
    return {
        "schema":"ukdale-logged-kettle-observed-power-windows-v1",
        "metadata_commit":COMMIT,"house":house,
        "meter":METERS[house],"role":ROLES[house],
        "original_source_file":station["original_file"],
        "original_meter_file_sha256":digest.hexdigest(),
        "original_raw_rows":n,"first_raw_unix_s":first,"last_raw_unix_s":last_seen,
        "accepted_complete_10min_windows":len(results),
        "source_meter_clip_limit_w":4000,
        "target":"arithmetic mean of logged nonimputed kettle plug power samples in W, not true time-integrated Wh",
        "source_sensor_available_at_known":False,
        "original_source_publisher_archive_integrity_verified":False,
        "new_model_fit_or_score_performed":False,
        "independent_generalization_evaluated":False,
        "recorded_windows":results}

def complete_slot(slot,samples):
    """Require near-complete genuine logged snapshots, not synthetic interpolation."""
    if len(samples)<MIN_SAMPLES:
        return None
    start=slot*600
    if samples[0][0]-start>15 or start+600-samples[-1][0]>15:
        return None
    if any(b[0]-a[0]>MAX_GAP_SECONDS for a,b in zip(samples,samples[1:])):
        return None
    return {"source_10min_window_start_unix_s":start,
            "source_10min_window_end_unix_s":start+600,
            "actual_source_log_sample_count":len(samples),
            "logged_sample_average_w":sum(w for _,w in samples)/len(samples)}

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--metadata-audit-only",action="store_true")
    p.add_argument("--original-channel",type=Path)
    p.add_argument("--house",type=int)
    p.add_argument("--metadata-receipt",type=Path)
    p.add_argument("--source-sha256")
    args=p.parse_args(argv)
    try:
        if args.out.exists():raise AdmissionError("refuse to overwrite original source receipt")
        if args.metadata_audit_only:
            if any(x is not None for x in (args.original_channel,args.house,args.metadata_receipt,args.source_sha256)):
                raise AdmissionError("metadata-only audit must not access any meter readings")
            result=audit_publisher_metadata()
        else:
            if (args.original_channel is None or args.house is None
                or args.metadata_receipt is None or args.source_sha256 is None):
                raise AdmissionError("original source, house, actual checksum and metadata required")
            if args.house==5:
                raise AdmissionError("sealed House 5 raw observations MUST NOT be opened")
            expected=Path(f"house_{args.house}")/f"channel_{METERS[args.house]}.dat" if args.house in METERS else None
            if args.original_channel.parts[-2:]!=expected.parts if expected is not None else True:
                raise AdmissionError("unexpected house/channel original meter path")
            with args.original_channel.open("rb") as f:
                result=aggregate_kettle_snapshots(f,args.house,
                    json.loads(args.metadata_receipt.read_bytes()),args.source_sha256)
        args.out.parent.mkdir(parents=True,exist_ok=True)
        args.out.write_text(json.dumps(result,sort_keys=True,indent=2)+"\n")
        print(json.dumps({"schema":result["schema"],
                          "source_role":"METADATA_ONLY" if args.metadata_audit_only else result["role"],
                          "trained_model_evaluated":False,
                          "sealed_house5_power_opened":False,
                          "source_status":"SOURCE_METADATA_ONLY" if args.metadata_audit_only else
                          "ORIGINAL_LOGGED_WINDOWS_ONLY_NOT_FORECAST_QUALIFICATION"},sort_keys=True))
        return 0
    except (AdmissionError,ValueError,TypeError,OSError,UnicodeError) as e:
        print(json.dumps({"status":"BLOCKED_OR_FAILED","reason":str(e)}),file=sys.stderr)
        return 2

if __name__=="__main__":
    raise SystemExit(main())
