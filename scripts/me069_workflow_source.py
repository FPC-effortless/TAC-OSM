"""Original BPI2012 source, case/time boundaries and FIT-only preprocessing."""
from __future__ import annotations
import gzip
import hashlib
import json
import math
from datetime import datetime,timezone
from pathlib import Path
from urllib.request import urlopen
import xml.etree.ElementTree as ET
import numpy as np

SOURCE_URL="https://ndownloader.figshare.com/files/24027287"
SOURCE_MD5="74c7ba9aba85bfcb181a22c9d565e5b5"
SOURCE_SHA256="5cd9cc16b9bcb20bd4aae45666a5d87479ddbf47e6371618b6ad217174cecdf3"
BYTE_IDENTICAL_MIRROR="https://media.githubusercontent.com/media/Gength/generalization_analysis/82d076ea0bb77c1164e6990b33b1f102540dacdc/data/BPI-Challenge_2012/BPI_Challenge_2012.xes.gz"
ROOT=Path("research/mathematical-engineering/me069")
HISTORY=32


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":")).encode()).hexdigest()


def write_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,sort_keys=True)+"\n")


def source_path(cache):
    path=Path(cache)/"BPI_Challenge_2012.xes.gz"
    if not path.exists():
        path.parent.mkdir(parents=True,exist_ok=True)
        try:
            with urlopen(SOURCE_URL,timeout=12) as response:raw=response.read(4_000_000)
        except OSError:
            with urlopen(BYTE_IDENTICAL_MIRROR,timeout=30) as response:raw=response.read(4_000_000)
        if len(raw)!=3342406 or hashlib.md5(raw).hexdigest()!=SOURCE_MD5:
            raise ValueError("publisher file identity mismatch")
        path.write_bytes(raw)
    raw=path.read_bytes()
    if (len(raw)!=3342406 or hashlib.md5(raw).hexdigest()!=SOURCE_MD5
            or hashlib.sha256(raw).hexdigest()!=SOURCE_SHA256):
        raise ValueError("cached original file mismatch")
    return path,hashlib.sha256(raw).hexdigest()


def parse_cases(path):
    cases=[]
    with gzip.open(path,"rb") as stream:
        for _,trace in ET.iterparse(stream,events=("end",)):
            if trace.tag.rsplit("}",1)[-1]!="trace":continue
            attrs={e.get("key"):e.get("value") for e in trace if e.tag.rsplit("}",1)[-1]!="event"}
            events=[]
            for event in trace:
                if event.tag.rsplit("}",1)[-1]!="event":continue
                a={e.get("key"):e.get("value") for e in event}
                timestamp=datetime.fromisoformat(a["time:timestamp"].replace("Z","+00:00")).timestamp()
                events.append([a["concept:name"]+"|"+a.get("lifecycle:transition",""),a.get("org:resource","UNKNOWN"),timestamp])
            if events:
                if any(events[i][2]<events[i-1][2] for i in range(1,len(events))):
                    raise ValueError("original event order is not chronological")
                cases.append({"id":attrs["concept:name"],"events":events})
            trace.clear()
    if len({c["id"] for c in cases})!=len(cases):raise ValueError("duplicate original case identity")
    return sorted(cases,key=lambda c:(c["events"][0][2],c["id"]))


def split_cases(cases):
    a,b=int(.6*len(cases)),int(.8*len(cases))
    first_dev,first_test=cases[a]["events"][0][2],cases[b]["events"][0][2]
    fit=[c for c in cases[:a] if c["events"][-1][2]<first_dev]
    dev=[c for c in cases[a:b] if c["events"][-1][2]<first_test]
    test=cases[b:]
    return {"FIT":fit,"DEV":dev,"TEST":test},{
        "FIT_label_availability_cutoff_UTC":datetime.fromtimestamp(first_dev,timezone.utc).isoformat(),
        "DEV_label_availability_cutoff_UTC":datetime.fromtimestamp(first_test,timezone.utc).isoformat(),
        "FIT_crossing_excluded":a-len(fit),"DEV_crossing_excluded":b-a-len(dev)}


def fit_preprocessing(fit):
    activities=sorted({e[0] for c in fit for e in c["events"]})
    resources=sorted({e[1] for c in fit for e in c["events"]})
    av={s:i+1 for i,s in enumerate(activities)};rv={s:i+1 for i,s in enumerate(resources)}
    n=len(av)+1;counts=np.full((n,n),.5,dtype=np.float64)
    delays=[[] for _ in range(n)];all_delays=[]
    for c in fit:
        for current,following in zip(c["events"],c["events"][1:]):
            i,j=av[current[0]],av[following[0]]
            dt=following[2]-current[2];counts[i,j]+=1;delays[i].append(dt);all_delays.append(dt)
    positive=np.asarray([x for x in all_delays if x>0],dtype=np.float64)
    if len(positive)==0:raise ValueError("no real positive waiting times")
    global_log=np.log(positive)
    priors=[]
    for row in delays:
        logs=np.log([x for x in row if x>0])
        prob=(sum(x>0 for x in row)+.5)/(len(row)+1) if row else (len(positive)+.5)/(len(all_delays)+1)
        priors.append([float(np.log(prob/(1-prob))),
            float(logs.mean() if len(logs) else global_log.mean()),
            float(np.log(max(.1,min(5.,logs.std() if len(logs)>1 else global_log.std()))))])
    return {"activities":av,"resources":rv,"transition_log_probability":np.log(counts/counts.sum(1,keepdims=True)).tolist(),
        "time_prior":priors,"long_wait_threshold_seconds":float(np.quantile(positive,.75)),
        "global_next_activity_probability":(counts.sum(0)/counts.sum()).tolist(),
        "global_wait_median_seconds":float(np.median(all_delays)),
        "global_wait_positive_probability":len(positive)/len(all_delays),
        "global_log_positive_mean":float(global_log.mean()),"global_log_positive_std":float(global_log.std()),
        "FIT_event_count":sum(len(c["events"]) for c in fit),"FIT_positive_gap_count":len(positive)}


def encode_cases(cases,pre):
    encoded=[]
    for case in cases:
        events=case["events"];times=np.asarray([e[2] for e in events],dtype=np.float64)
        gaps=np.r_[0.,np.diff(times)]
        f=[]
        for i,t in enumerate(times):
            date=datetime.fromtimestamp(t,timezone.utc);hour=date.hour+date.minute/60.
            f.append([math.log1p(gaps[i])/12.,math.log1p(t-times[0])/16.,
                math.sin(2*math.pi*hour/24),math.cos(2*math.pi*hour/24),
                math.sin(2*math.pi*date.weekday()/7),math.cos(2*math.pi*date.weekday()/7),
                math.log1p(i)/5.,float(i>0 and events[i][0]==events[i-1][0])])
        encoded.append({"id":case["id"],"activity":np.asarray([pre["activities"].get(e[0],0) for e in events],dtype=np.int64),
            "resource":np.asarray([pre["resources"].get(e[1],0) for e in events],dtype=np.int64),
            "features":np.asarray(f,dtype=np.float32),"gap":gaps.astype(np.float32),
            "timestamps":times,"n":len(events)})
    return encoded


def load(cache):
    path,sha=source_path(cache);cases=parse_cases(path);splits,cutoffs=split_cases(cases)
    pre=fit_preprocessing(splits["FIT"])
    return {key:encode_cases(rows,pre) for key,rows in splits.items()},pre,{
        "source_sha256":sha,"publisher_md5":SOURCE_MD5,"original_cases":len(cases),
        "original_events":sum(len(c["events"]) for c in cases),"cutoffs":cutoffs,
        "cohorts":{key:{"cases":len(rows),"events":sum(len(c["events"]) for c in rows),
            "ids_sha256":digest([c["id"] for c in rows])} for key,rows in splits.items()},
        "FIT_preprocessing_sha256":digest(pre),"heldout_access":"original file parsed for case/time partition; no heldout performance scored at qualification"}


if __name__=="__main__":
    import argparse
    p=argparse.ArgumentParser();p.add_argument("--cache",type=Path,required=True);a=p.parse_args()
    _,pre,receipt=load(a.cache);write_json(ROOT/"SOURCE_QUALIFICATION.json",receipt)
    print(json.dumps(receipt,indent=2));print("FIT_vocabulary",len(pre["activities"]),len(pre["resources"]))
