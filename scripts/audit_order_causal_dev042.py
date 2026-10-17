"""DEV-042 separate deterministic generator and physical-results audit."""
from __future__ import annotations
import argparse
import json
from pathlib import Path

from tac_osm.development.order_causal_cases import make_order_pair,last_observations
from tac_osm.development.schema_maintenance import execute,exact_public_targets


def source_cases():
    out={}
    for delay in (32,128,512):
        for deadline in ("before","at","after"):
            for rep in range(16):
                pair=make_order_pair(420000+rep,512,delay,deadline)
                assert last_observations(pair[0].observations)==last_observations(pair[1].observations)
                if deadline!="before":
                    assert pair[0].wait_targets[pair[0].target_entity]!=pair[1].wait_targets[pair[1].target_entity]
                for case in pair:
                    assert exact_public_targets(case.observations,case.tick,case.horizon)==case.wait_targets
                    assert case.fingerprint not in out
                    out[case.fingerprint]=case
    assert len(out)==288
    return out


def audit(report,archive,rows,threshold):
    seen=set()
    totals=dict(failures=0,unnecessary_services=0,services=0,total_cost=0)
    correct_mature=0;mature=0;correct_pending=0;pending=0
    for row in rows:
        fingerprint=row["fingerprint"];assert fingerprint in archive and fingerprint not in seen
        seen.add(fingerprint)
        case=archive[fingerprint]
        assert row["target_entity"]==case.target_entity and row["deadline"] in ("before","at","after")
        assert row["labels"]==[[case.wait_targets[e],case.service_targets[e],case.current_targets[e]] for e in range(8)]
        probs=row["probabilities"]
        assert len(probs)==8 and all(len(p)==3 and all(0<=v<=1 for v in p) for p in probs)
        chosen=tuple(e for e,pp in enumerate(probs) if pp[0]-pp[1]>threshold)
        physical=execute(case.world,case.horizon,chosen,0)
        assert {k:row["physical"][k] for k in ("failures","unnecessary_services","services","total_cost")}=={
            "failures":physical["failures"],"unnecessary_services":physical["unnecessary_services"],
            "services":len(chosen),"total_cost":physical["total_cost"]}
        for k in totals:totals[k]+=row["physical"][k]
        target=case.target_entity
        good=int((probs[target][0]>.5)==bool(case.wait_targets[target]))
        if row["deadline"]=="before":pending+=1;correct_pending+=good
        else:mature+=1;correct_mature+=good
    assert seen==set(archive) and mature==192 and pending==96
    saved=report["summary"]
    assert saved["mature_worlds"]==mature and saved["worlds"]==mature+pending
    assert abs(saved["mature_wait"]-correct_mature/mature)<1e-12
    assert abs(saved["pending_wait"]-correct_pending/pending)<1e-12
    assert saved["physical"]==totals
    assert abs(saved["cost_per_entity"]-totals["total_cost"]/(len(seen)*8))<1e-12
    return dict(mature_wait=saved["mature_wait"],pending_wait=saved["pending_wait"],
                cost_per_entity=saved["cost_per_entity"],failures=totals["failures"],
                unnecessary_services=totals["unnecessary_services"])


def run(paths,output):
    cases=source_cases();parts=[]
    for path in paths:
        path=Path(path)
        r=json.loads(path.read_text())
        assert r["study"]=="DEV-042" and r["status"]=="ADAPTIVE_DEVELOPMENT_NOT_CONFIRMATORY"
        assert r["model_source"]=="4ba2ea65cd3d4f3961277d80c49d7fb641f556fc"
        arms={}
        for name,details in r["arms"].items():
            assert name in ("bounded_trainable","bounded_random_frozen","memoryless")
            arms[name]=audit(details,cases,details["rows"],details["threshold"])
        last=audit(r["last_event"],cases,r["last_event"]["rows"],r["last_event"]["threshold"])
        exact=r["exact_public_history"]["summary"]
        assert exact["mature_wait"]==1 and exact["physical"]["failures"]==0
        assert abs(last["mature_wait"]-.5)<1e-12
        # A last-event-only deterministic rule must assign identical actions to partners.
        lr=r["last_event"]["rows"]
        for i in range(0,len(lr),2):
            assert lr[i]["probabilities"]==lr[i+1]["probabilities"]
        parts.append(dict(seed=r["seed"],arms=arms,last_event=last,exact_public=exact))
    assert {p["seed"] for p in parts}=={4101,4102,4103}
    out=dict(study="DEV-042",status="AUDITED_TRANSFER_DIAGNOSTIC_NOT_CONFIRMATORY",
             per_seed=parts,last_event_mature_ceiling=.5,
             no_new_training=True,architecture_superiority_established=False)
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    (output/"analysis.json").write_text(json.dumps(out,indent=2,allow_nan=False)+"\n")
    print(json.dumps(out,indent=2),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("reports",nargs=3)
    p.add_argument("--output",required=True)
    a=p.parse_args();run(a.reports,a.output)
