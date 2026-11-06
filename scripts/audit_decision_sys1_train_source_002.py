#!/usr/bin/env python3
"""TRAIN-only offline file audit. No networks, raw uploads, model, optimizer or TEST.

The original CSV bytes must already be acquired from the publisher. A matching
historical hash is provenance evidence, not independent signed training authority.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
from tac_osm.decision_sys1_data_002 import Readout, FEATURES, split_of, survival_target

READOUT_SHA256 = 'e01cb0bd87dfab4c9dbad215d51d81282fc0d413be96d6f819ea872fb7a3c715'


def sha256(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def dump(path, content):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(content,sort_keys=True,indent=2,allow_nan=False)+'\n')
    temp.replace(path)


def audit(readouts, tte, output, expected_readout_sha=READOUT_SHA256):
    readouts,tte,output=Path(readouts),Path(tte),Path(output)
    if readouts.name!='train_operational_readouts.csv' or tte.name!='train_tte.csv':
        raise ValueError('only the two exact publisher TRAIN filenames are admitted')
    actual=sha256(readouts)
    if actual!=expected_readout_sha:
        raise ValueError('readout bytes do not match the pinned original-source hash')
    counts={};last={};examples=hashlib.sha256();rows=0
    with readouts.open(newline='') as f:
        for raw in csv.DictReader(f):
            r=Readout.from_mapping(raw)
            if r.vehicle_id in last and r.time_step<=last[r.vehicle_id]:
                raise ValueError('duplicate or reverse per-vehicle time')
            last[r.vehicle_id]=r.time_step
            counts[r.vehicle_id]=counts.get(r.vehicle_id,0)+1;rows+=1
            examples.update((r.example_id+'\n').encode())
    if not rows:raise ValueError('empty readout source')
    # Must be persisted BEFORE the first label file open. Never claim a new
    # scientific holdout: previous ME001B read this publisher training source.
    split={'source_sha256':actual,'rows':rows,'example_order_sha256':examples.hexdigest(),
           'fit_ids':sorted(v for v in counts if split_of(v)=='fit'),
           'dev_ids':sorted(v for v in counts if split_of(v)=='dev'),
           'status':'TRAIN_INTERNAL_DEVELOPMENT_PREVIOUSLY_EXPOSED'}
    dump(output/'split_before_labels.json',split)
    targets={}
    with tte.open(newline='') as f:
        reader=csv.DictReader(f)
        if set(reader.fieldnames or [])!={'vehicle_id','length_of_study_time_step','in_study_repair'}:
            raise ValueError('unexpected TRAIN outcome schema')
        for r in reader:
            key=r['vehicle_id']
            if key in targets:raise ValueError('duplicate outcome vehicle')
            repair=int(r['in_study_repair']);end=float(r['length_of_study_time_step'])
            survival_target(0.,end,repair)  # Validate positive lifespan and indicator.
            targets[key]=(end,repair)
    if set(targets)!=set(counts):raise ValueError('vehicle join mismatch')
    statuses={};excluded=0
    with readouts.open(newline='') as f:
        for raw in csv.DictReader(f):
            r=Readout.from_mapping(raw);end,repair=targets[r.vehicle_id]
            if r.time_step>end or (repair==1 and r.time_step==end):
                excluded+=1;continue
            status=survival_target(r.time_step,end,repair).status
            statuses[status]=statuses.get(status,0)+1
    report={'study':'DECISION-SYS1-002','stage':'SOURCE_AUDIT_ONLY',
            'readouts_sha256':actual,'tte_sha256':sha256(tte),'rows':rows,
            'vehicles':len(counts),'features':len(FEATURES),'status_counts':statuses,
            'excluded_at_or_after_repair_or_after_followup':excluded,
            'split_manifest_sha256':sha256(output/'split_before_labels.json'),
            'real_model_predictions':0,'optimizer_steps':0,
            'test_opened':False,'validation_opened':False,'training_admitted':False,
            'remaining':['publisher exact-event boundary alignment',
                         'original outcome-source attestation','qualified Mamba-2 and equal budgets',
                         'signed evaluator/parity and compute admission','full source/math correspondence']}
    dump(output/'source_audit.json',report)
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--readouts',required=True,type=Path)
    p.add_argument('--tte',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path)
    a=p.parse_args();print(json.dumps(audit(a.readouts,a.tte,a.output),indent=2))
