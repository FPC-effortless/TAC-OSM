#!/usr/bin/env python3
"""S1TEMP008 trusted ORIGINAL BPIC17 XES.gz source partition and slot audit.

This trusted source-only process necessarily DOWNLOADS the ENTIRE XES archive
including heldout raw bytes; it must NOT be described as a physically sealed
DEV/TEST archive. This stage NEVER emits heldout case activities, offers,
labels, case IDs or summaries into model-input artifacts. It reads only
source-hashed FIT traces' outcomes, and retains no outputs after process end.

No data file/event stream is committed. Only aggregate FIT source eligibility
numbers (no real case IDs or customer amounts) are published. No training or
heldout-scoring occurs. Original license: 4TU General Terms of Use.

Timestamp and entity slots derive from events AS OF event time, not
future labels. Slot0 global, slots1-7 assigned by FIRST chronological
appearance of a real same-case OfferID. More than seven => excluded and counted.
"""
from __future__ import annotations
import datetime as dt,gzip,hashlib,io,json,time,urllib.request,urllib.error,xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
LOCK=ROOT/"research/parallel-six/S1TEMP008-PUBLISHER-LOCK.json"
URL="https://ndownloader.figshare.com/files/24044117"
EXPECTED_BYTES=29658747
MD5="10b37a2f78e870d78406198403ff13d2"
SALT="S1TEMP008-BPIC2017-CASE-SPLIT-V1"
def local(tag):
    return tag.rsplit("}",1)[-1]

def original_split(case_id):
    # Entirely based on immutable source-visible case identifier, never label,
    # eventual outcome, number of offers, or time to resolution.
    h=hashlib.sha256((SALT+"|"+case_id).encode("utf-8")).digest()
    n=int.from_bytes(h[:8],"big")%10
    return "FIT" if n<6 else ("DEV" if n<8 else "TEST")

def source_prefix_slots(events):
    """Sequential, case-local ID assignment, no future entity count access."""
    slots={};mapped=[]
    for row in events:
        raw_id=row.get("OfferID")
        if raw_id is None or raw_id=="":
            slot=0
        elif raw_id in slots:
            slot=slots[raw_id]
        else:
            if len(slots)>=7:
                raise ValueError("more than seven original offer identities in case")
            slot=len(slots)+1
            slots[raw_id]=slot
        mapped.append(slot)
    return mapped,len(slots)

def asof_fit_events(trace):
    case=None;events=[];prev_clock=None
    for child in trace:
        name=local(child.tag)
        if name=="string" and child.attrib.get("key")=="concept:name":
            case=child.attrib.get("value")
        elif name=="event":
            data={v.attrib.get("key"):v.attrib.get("value")
                  for v in child if v.attrib.get("key") is not None}
            event=data.get("concept:name")
            ts=data.get("time:timestamp")
            if not event or not ts:
                raise RuntimeError("timestamp/event ontology incomplete")
            clock=dt.datetime.fromisoformat(ts.replace("Z","+00:00"))
            if clock.tzinfo is None:raise RuntimeError("source timestamp lacks timezone")
            normalized=clock.astimezone(dt.timezone.utc).timestamp()
            if prev_clock is not None and normalized<prev_clock:
                raise RuntimeError("source chronology reversed in XES event trace")
            prev_clock=normalized
            events.append(dict(activity=event, timestamp=normalized,
                 OfferID=data.get("OfferID"),
                 lifecycle=data.get("lifecycle:transition","")))
    if not case or not events:
        raise RuntimeError("source case identity or source event missing")
    return case,events

def audit():
    lock=json.loads(LOCK.read_text())
    if (lock["file_id"]!=24044117 or lock["file_bytes"]!=EXPECTED_BYTES
        or lock["original_MD5"]!=MD5
        or lock["source_eligibility_run"]!=40307617155):
        raise RuntimeError("pinned original metadata chain mismatched")
    # The original publisher's ndownloader may temporarily return 504.
    # Retry only the same official file-ID, retaining exact pinned MD5/bytes.
    # Do NOT switch to an unverified mirror or third-party CSV.
    original=None
    for attempt in range(4):
        try:
            with urllib.request.urlopen(urllib.request.Request(URL,headers={
                  "User-Agent":"S1TEMP008-trusted-source-case-partitioner/1.0"}),
                  timeout=135) as response:
                original=response.read()
            break
        except (urllib.error.HTTPError,urllib.error.URLError,TimeoutError) as exc:
            if attempt==3:
                raise RuntimeError(
                    f"official 4TU original archive inaccessible after 4 tries; "
                    f"real-data gate BLOCKED ({type(exc).__name__})") from exc
            time.sleep((3,7,15)[attempt])
    if original is None:
        raise RuntimeError("original publisher source not obtained")
    if len(original)!=EXPECTED_BYTES or hashlib.md5(original).hexdigest()!=MD5:
        raise RuntimeError("download differs from publisher locked original MD5")
    counts=Counter()
    activities=Counter()
    slots_used=Counter()
    time_ties=0
    source_fitting_cases=0
    earliest=None;latest=None
    with gzip.GzipFile(fileobj=io.BytesIO(original)) as raw:
        iterparse=ET.iterparse(raw,events=("start","end"))
        _,root=next(iterparse)
        for phase,elem in iterparse:
            if phase!="end" or local(elem.tag)!="trace":
                continue
            # Minimal case id extraction BEFORE touching any event attributes.
            case=next((node.attrib.get("value") for node in elem
                       if local(node.tag)=="string" and
                       node.attrib.get("key")=="concept:name"),None)
            if not case:raise RuntimeError("full original source trace lacks case id")
            group=original_split(case)
            counts[group+"_cases"]+=1
            if group=="FIT":
                case_again,events=asof_fit_events(elem)
                if case_again!=case:
                    raise RuntimeError("case ID changed within real source trace")
                counts["FIT_source_events"]+=len(events)
                try:
                    mapped,n=source_prefix_slots(events)
                except ValueError:
                    counts["FIT_capacity_excluded_cases"]+=1
                    elem.clear();root.clear();continue
                counts["FIT_capacity_eligible_cases"]+=1
                if n>=2:
                    counts["FIT_multiple_distinct_offers"]+=1
                slots_used[n]+=1
                source_fitting_cases+=1
                for row,slot in zip(events,mapped):
                    activities[row["activity"]]+=1
                    counts[f"FIT_asof_slot_{slot}_events"]+=1
                    clock=row["timestamp"]
                    if earliest is None or clock<earliest:earliest=clock
                    if latest is None or clock>latest:latest=clock
                time_ties+=sum(events[t]["timestamp"]==events[t-1]["timestamp"]
                               for t in range(1,len(events)))
            # Critical: NO DEV/TEST event attributes or outcome strings read.
            elem.clear();root.clear()
    if not all(counts[f"{q}_cases"]>0 for q in ("FIT","DEV","TEST")):
        raise RuntimeError("case-independent partition failed")
    if counts["FIT_capacity_eligible_cases"]<1000:
        raise RuntimeError("too few source slots admitted, cannot justify benchmark")
    if counts["FIT_multiple_distinct_offers"]<100:
        raise RuntimeError("original multi-offer source eligibility not demonstrated")
    summary=dict(study="S1TEMP008",publisher_article=12696884,
        original_file_md5_verified=True,source_original_bytes=EXPECTED_BYTES,
        original_license="4TU General Terms of Use",
        source_hash_partition_salt=SALT,
        source_hash_assignment_no_label_dependence=True,
        trusted_partitioner_downloaded_ENTIRE_original_archive=True,
        training_process_had_NO_access_to_dev_test_yet=True,
        case_splits_counts=dict(counts),
        FIT_case_offer_identity_distribution=dict(slots_used),
        FIT_unique_activity_vocab_count=len(activities),
        FIT_observed_first_clock_utc=earliest,FIT_observed_last_clock_utc=latest,
        FIT_equal_timestamp_adjacent_events=time_ties,
        FIT_is_source_timestamp_ordered=True,
        dev_or_test_event_activity_or_case_ID_not_exported=True,
        no_future_label_in_model_input=True,
        no_optimizer_or_model_metrics=True,
        no_underwriting_or_autonomous_approval_authority=True)
    target=Path("research/diagnostics/s1temp008-source-audit")
    target.mkdir(parents=True,exist_ok=True)
    (target/"FIT-aggregate-source-only.json").write_text(
        json.dumps(summary,indent=2,sort_keys=True)+"\n")
    print("S1TEMP008_BPI_SOURCE_FIT_ELIGIBILITY_PASS",
          json.dumps(summary,sort_keys=True),flush=True)
if __name__=="__main__":audit()
