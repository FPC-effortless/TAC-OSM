"""PAR-MEM-014 EPA AQS original 2024 PM2.5 metadata-only source admission.

First public source ingestion is NOT neural training/evaluation and does not
access any measured concentration, MDL, instrument uncertainty, latitude or
longitude as a Python value.  It reads ONLY owner, GMT timestamp, method,
qualifier and unit metadata from the SHA-pinned publisher ZIP.
No label, future concentration distribution, threshold, site selection by
target values, historical retrieval MAE or model prediction is computed.

Required original EPA ZIP:
  https://aqs.epa.gov/aqsweb/airdata/hourly_88101_2024.zip

Original dataset size ~8.1M hourly rows: stream one CSV member from ZIP,
without materializing CSV in memory. Eligible monitor metadata is selected
by preregistered SHA256 owner hash and calendar phase, no target-driven
selection. Unknown units/qualifiers and excessive duplicates fail scientific
admission, not silently accepted as training source quality.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from hashlib import sha256
import io
import json
from pathlib import Path
import re
import zipfile

ROOT=Path(__file__).resolve().parents[1]
PROTOCOL=ROOT/"research/parallel-six/PAR-MEM-014-PROTOCOL.json"
RECEIPT=ROOT/"research/source-bundles/PAR-MEM-014/EPA2024-METADATA-RECEIPT.json"
YEAR="2024"
ZIP_NAME="hourly_88101_2024.zip"
COLS=(
    "State Code","County Code","Site Num","Parameter Code","POC",
    "Date GMT","Time GMT","Units of Measure","Qualifier",
    "Method Type","Method Code","Date of Last Change"
)
# Forbidden original publisher numerical fields are INTENTIONALLY NOT indexed:
# Sample Measurement, MDL, Uncertainty, Latitude, Longitude.

def checked_protocol():
    cfg=json.loads(PROTOCOL.read_text())
    expected={
        "study_id":"PAR-MEM-014-EPA-AQS-2024-INDEPENDENT-SOURCE-ADMISSION",
        "status":"PUBLISHER_ORIGINAL_METADATA_ONLY_INTAKE_NO_TARGET_VALUE_ACCESS",
        "publisher_parameter_code":"88101",
        "publisher_year":"2024",
        "minimum_source_hour_metadata_count_per_monitor_in_its_assigned_phase":160,
        "no_new_model_family":False,
    }
    for k,v in expected.items():
        if cfg.get(k)!=v:
            raise RuntimeError("registered original EPA metadata gate changed: "+k)
    if cfg["training_compare_budget"]!="equal actual source observations/target splits/optimizer budgets, complete forward/backward/optimizer mathematical FLOPs; CPU time need NOT match":
        raise RuntimeError("compulsory full learned compare budget silently removed")
    if cfg["forbidden_numeric_fields_during_intake"]!=[
            "Sample Measurement","MDL","Uncertainty","Latitude","Longitude"]:
        raise RuntimeError("protected EPA measured numeric column gate changed")
    return cfg

def source_sha(path):
    digest=sha256()
    with path.open("rb") as handle:
        for block in iter(lambda:handle.read(1048576),b""):
            digest.update(block)
    return digest.hexdigest()

def source_owner(row):
    if not (row["State Code"].isdigit() and row["County Code"].isdigit() and
            row["Site Num"].isdigit() and row["POC"].isdigit() and
            row["Parameter Code"]=="88101"):
        raise ValueError("invalid EPA monitor owner/parameter code")
    state=row["State Code"].zfill(2)
    county=row["County Code"].zfill(3)
    site=row["Site Num"].zfill(4)
    poc=str(int(row["POC"]))
    if len(state)!=2 or len(county)!=3 or len(site)!=4:
        raise ValueError("out-of-range original EPA source owner FIPS identity")
    return f"{state}-{county}-{site}-88101-{poc}"

def source_phase(owner, date, cfg):
    slot=int(sha256(owner.encode()).hexdigest()[:16],16)%100
    if slot<70: phase="train"
    elif slot<85: phase="dev"
    else: phase="sealed_test"
    start,end=cfg["temporal_split_utc"][phase]
    return phase if start<=date<end else None

def clean_source_hour(row):
    day=row["Date GMT"]
    clock=row["Time GMT"]
    if not (re.fullmatch(r"2024-\d\d-\d\d",day) and
            re.fullmatch(r"(?:[01]\d|2[0-3]):[0-5]\d",clock)):
        raise ValueError("original source hourly GMT timestamp invalid")
    if clock[-2:]!="00":
        raise ValueError("EPA original hourly sample must begin at clock hour")
    return day+"T"+clock+":00Z"

def assess_archive(path,cfg):
    zip_sha=source_sha(path)
    if not zipfile.is_zipfile(path):
        raise RuntimeError("publisher original ZIP is not ZIP")
    with zipfile.ZipFile(path) as archive:
        members=[z for z in archive.infolist() if z.filename.lower().endswith(".csv")]
        if len(members)!=1:
            raise RuntimeError("expected ONE publisher-original EPA hourly CSV member")
        member=members[0]
        if member.file_size>6_000_000_000 or member.file_size<10_000_000:
            raise RuntimeError("unexpected publisher original source CSV length")
        with archive.open(member,"r") as binary:
            with io.TextIOWrapper(binary,encoding="utf-8-sig",newline="") as text_input:
                reader=csv.reader(text_input)
                headers=next(reader)
                lookup={s.strip():i for i,s in enumerate(headers)}
                required=set(COLS)
                if not required.issubset(lookup):
                    raise RuntimeError("publisher original EPA hourly header mismatch "+str(
                        sorted(required-set(lookup))))
                if not set(cfg["forbidden_numeric_fields_during_intake"]).issubset(lookup):
                    raise RuntimeError("EPA original measured target/quantities header not present")
                # This deliberately NEVER indexes any column outside COLS.
                # Only one metadata field is scanned per row, and the full
                # numeric physical target value never becomes an object.
                metadata_indexes=[lookup[c] for c in COLS]
                overall=Counter()
                phase_rows=Counter()
                units=Counter()
                qualifiers=Counter()
                method_types=Counter()
                methods=defaultdict(set)
                owner_timestamps=defaultdict(set)
                duplicate_timestamps=defaultdict(set)
                observed_phase_counts=defaultdict(Counter)
                invalid_metadata=Counter()
                for line_no,values in enumerate(reader,2):
                    overall["publisher_raw_data_rows"]+=1
                    if len(values)!=len(headers):
                        invalid_metadata["noncanonical_field_count"]+=1
                        continue
                    row=dict(zip(COLS,(values[i].strip()
                                       for i in metadata_indexes)))
                    try:
                        owner=source_owner(row)
                        hour=clean_source_hour(row)
                    except ValueError:
                        invalid_metadata["noncanonical_owner_or_source_clock"]+=1
                        continue
                    phase=source_phase(owner,row["Date GMT"],cfg)
                    if phase is None:
                        continue
                    phase_rows[phase]+=1
                    unit=row["Units of Measure"]
                    qualifier=row["Qualifier"]
                    method=row["Method Code"]
                    units[(phase,unit)]+=1
                    qualifiers[(phase,qualifier)]+=1
                    method_types[(phase,row["Method Type"])]+=1
                    methods[(phase,owner)].add(method)
                    index=(phase,owner)
                    timestamps=owner_timestamps[index]
                    if hour in timestamps:
                        duplicate_timestamps[index].add(hour)
                    else:
                        timestamps.add(hour)
                    if overall["publisher_raw_data_rows"]%1_000_000==0:
                        print("PAR_MEM_014_METADATA_ONLY_PUBLISHER_ROWS="+str(
                            overall["publisher_raw_data_rows"]),flush=True)
                monitors_by_phase={}
                monitor_counts={}
                collision_count=Counter()
                phase_summary={}
                for phase in ("train","dev","sealed_test"):
                    eligible=[]
                    counts=[]
                    for (role,owner),clocks in owner_timestamps.items():
                        if role!=phase:
                            continue
                        duplicates=duplicate_timestamps[(phase,owner)]
                        valid_clock_count=len(clocks)-len(duplicates)
                        collision_count[phase]+=len(duplicates)
                        if valid_clock_count>=cfg[
                             "minimum_source_hour_metadata_count_per_monitor_in_its_assigned_phase"]:
                            eligible.append(owner)
                            counts.append(valid_clock_count)
                    eligible=sorted(eligible)
                    monitors_by_phase[phase]=eligible
                    monitor_counts[phase]={
                        owner:len(owner_timestamps[(phase,owner)])-
                              len(duplicate_timestamps[(phase,owner)])
                        for owner in eligible
                    }
                    phase_summary[phase]=dict(
                        metadata_only_source_rows_seen=phase_rows[phase],
                        source_monitor_IDs_seen=sum(1 for x in owner_timestamps if x[0]==phase),
                        distinct_monitor_IDs_eligible=len(eligible),
                        total_distinct_valid_clock_hours_on_eligible_monitors=sum(counts),
                        original_source_hour_duplicate_collisions=collision_count[phase])
                for role in ("train","dev","sealed_test"):
                    if not monitors_by_phase[role]:
                        raise RuntimeError("EPA source lacks required 129-event monitor cohort in "+role)
                if (set(monitors_by_phase["train"]) & set(monitors_by_phase["dev"]) or
                    set(monitors_by_phase["train"]) & set(monitors_by_phase["sealed_test"]) or
                    set(monitors_by_phase["dev"]) & set(monitors_by_phase["sealed_test"])):
                    raise RuntimeError("EPA original source train/dev/TEST monitor owners overlapped")
                if overall["publisher_raw_data_rows"]<1_000_000:
                    raise RuntimeError("unexpectedly small EPA publisher national hourly archive")
                return dict(
                    study_id=cfg["study_id"],status="METADATA_ONLY_SOURCE_IDENTITY_RECEIPT",
                    downloaded_publisher_url=cfg["publisher_primary_data_url"],
                    sha256_original_publisher_zip=zip_sha,
                    original_zip_member_name=member.filename,
                    original_zip_member_uncompressed_bytes=member.file_size,
                    original_zip_member_crc32=member.CRC,
                    original_zip_compressed_bytes=path.stat().st_size,
                    original_csv_field_names=headers,
                    original_publisher_file_as_of=cfg["publisher_catalog_as_of"],
                    total_original_csv_data_rows=overall["publisher_raw_data_rows"],
                    invalid_source_metadata_rows=dict(sorted(invalid_metadata.items())),
                    phase_source_metadata_summary=phase_summary,
                    monitoring_unit_metadata_frequencies=[
                        dict(phase=phase,unit=u,rows=count)
                        for (phase,u),count in sorted(units.items())],
                    original_qualifier_metadata_frequencies=[
                        dict(phase=phase,qualifier=q,rows=count)
                        for (phase,q),count in sorted(qualifiers.items())],
                    original_method_type_metadata_frequencies=[
                        dict(phase=phase,method_type=m,rows=count)
                        for (phase,m),count in sorted(method_types.items())],
                    count_of_original_method_codes_by_owner=[
                        dict(phase=phase,monitor=owner,count=len(values))
                        for (phase,owner),values in sorted(methods.items())],
                    original_eligible_monitor_owner_IDs=monitors_by_phase,
                    eligible_owner_unique_nonduplicated_clock_counts=monitor_counts,
                    sealed_source_target_measurements_read=False,
                    physical_target_concentration_field_ever_accessed=False,
                    no_torch_or_new_training=True,
                    no_source_label_statistics_used_for_site_selection=True,
                    independent_heldout_numeric_evaluation_opened=False,
                    caller_did_not_require_EPA_API_key=True,
                    model_accuracy_claimed=False)

def run(archive,mode):
    cfg=checked_protocol()
    if mode not in ("intake","replay"):
        raise ValueError("mode must be intake or replay")
    if not archive.is_file() or archive.name!=ZIP_NAME:
        raise RuntimeError("pass actual publisher-original expected EPA zip path")
    if (mode=="intake" and RECEIPT.exists()) or (
        mode=="replay" and not RECEIPT.exists()):
        raise RuntimeError("original source receipt is write-ONCE, replay thereafter")
    report=assess_archive(archive,cfg)
    if mode=="intake":
        RECEIPT.parent.mkdir(parents=True,exist_ok=True)
        RECEIPT.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    else:
        known=json.loads(RECEIPT.read_text())
        if report!=known:
            raise RuntimeError("publisher-original SHA, source identity, time, or metadata changed")
    print("PAR_MEM_014_PUBLISHER_METADATA_ONLY_SOURCE_ADMISSION="+json.dumps({
        "mode":mode,
        "original_publisher_sha256":report["sha256_original_publisher_zip"],
        "original_source_rows":report["total_original_csv_data_rows"],
        "monitor_counts":{phase:record["distinct_monitor_IDs_eligible"]
                          for phase,record in report["phase_source_metadata_summary"].items()},
        "all_numeric_target_values_unread":True,
        "independent_TEST_labels_untouched":True,
        "no_training":True},sort_keys=True),flush=True)

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--source",type=Path,required=True)
    parser.add_argument("--mode",choices=("intake","replay"),required=True)
    config=parser.parse_args()
    run(config.source,config.mode)
