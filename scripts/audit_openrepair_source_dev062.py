"""Verify a pinned public repair-outcome source, without fitting or task claims."""
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path


def main(directory,output):
    root=Path(directory);pointer=(root/'pointer.txt').read_text().splitlines();expected=next(s.split(':',1)[1] for s in pointer if s.startswith('oid sha256:'));size=int(next(s.split()[1] for s in pointer if s.startswith('size ')));path=root/'repair-attempts.csv'
    with path.open('rb') as f:actual=hashlib.file_digest(f,'sha256').hexdigest()
    assert actual==expected and path.stat().st_size==size,'Not the complete pinned Git LFS object'
    outcomes=Counter();providers=Counter();dates=[];rows=0;ids=set();duplicates=0
    with path.open(newline='',encoding='utf-8-sig') as f:
        reader=csv.DictReader(f);fields=reader.fieldnames
        for r in reader:
            rows+=1;outcomes[r['repair_status']]+=1;providers[r['data_provider']]+=1
            if r['event_date']:dates.append(r['event_date'])
            duplicates+=r['id'] in ids;ids.add(r['id'])
    schema=json.loads((root/'tableschema.json').read_text());result={'status':'COMPLETE_SOURCE_AUDIT_ONLY','source':'https://github.com/openrepair/data','commit':(root/'commit.txt').read_text().strip(),'object':'OpenRepairData_v0.3_aggregate_202607_eee.csv','sha256':actual,'bytes':size,'rows':rows,'duplicate_ids':duplicates,'fields':fields,'reported_repair_status':dict(outcomes),'providers':dict(providers),'event_dates_lexical_range':[min(dates),max(dates)],'licenses':schema['licenses'],'meaning':'Actual community repair-attempt reports, not industrial inspection sequences or independent physical post-repair verification.','label_contract':{'Fixed':'Reported repair succeeded; no independent follow-up health measure','End of life':'Reported end-of-life result, not an interchangeable randomized alternative repair','Repairable':'Not a verified successful completed repair; preserve separately','Unknown':'Unknown outcome; never negative by default'},'leakage_exclusions':['repair_status is a target, never an input','repair_barrier_if_end_of_life is outcome-conditioned','problem explicitly describes fault AND SOLUTION; not certified pre-action text'],'missing_commercial_fields':['available action alternatives','inspection result and time sequence','intervention choice and causal assignment','actual inspection/repair cost','independent post-repair physical outcome and follow-up'],'model_fits':0,'commercial_validation':'NOT_ESTABLISHED'}
    out=Path(output);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:result[k] for k in ('status','rows','bytes','sha256','reported_repair_status','commercial_validation')}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--directory',required=True);p.add_argument('--output',required=True);a=p.parse_args();main(a.directory,a.output)
