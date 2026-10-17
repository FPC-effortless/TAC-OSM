"""DEV-048A full candidate replay and independent simulator cost checks."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/"src"),str(ROOT/"scripts")]
import torch
from run_verified_routing_dev048 import cohort, run_policy, summarize, choose, CANDIDATES, PROTOCOL
from run_frozen_policy_dev045 import digest
from tac_osm.development.scalable_maintenance import execute


def audit(folder):
    folder = Path(folder)
    r = json.loads((folder/"report.json").read_text())
    assert r["study"]=="DEV-048A" and not r["smoke"]
    assert r["protocol_sha256"]==digest(PROTOCOL)
    assert r["checkpoint_sha256"]==digest(folder/"router-model.pt")
    assert r["selection_lock_sha256"]==digest(folder/"selection-lock.json")
    selection = json.loads((folder/"selection-lock.json").read_text())
    assert selection==r["selection"] and choose(r["development"])[0]==selection["selected"]
    state = torch.load(folder/"router-model.pt",weights_only=True,map_location="cpu")
    split_fps = []
    for split in ("development","evaluation"):
        cases_conditions = list(cohort(r["seed"],split))
        split_fps.append({c.fingerprint for cs,_ in cases_conditions for c in cs})
        assert set(r[split])==set(CANDIDATES)
        for candidate,data in r[split].items():
            assert len(data["rows"])==len(cases_conditions)
            for saved,(cases,condition) in zip(data["rows"],cases_conditions):
                replay = dict(condition=condition,**run_policy(state,cases,candidate))
                # Runtime is hardware/load dependent; all logical outcomes and
                # operation counts must reproduce exactly.
                assert {k:v for k,v in replay.items() if k!="times"}=={k:v for k,v in saved.items() if k!="times"}
                if saved["physical"] is None:
                    assert saved["coverage"]<1
                    continue
                for case,actions,physical in zip(cases,saved["actions"],saved["physical"]):
                    services = tuple(e for e,a in enumerate(actions) if a==1)
                    inspection_count = len(case.entity_ids)
                    if candidate.endswith("_inspect"):
                        inspection_count += len(case.entity_ids)-int(candidate.split("_")[0].removeprefix("route"))
                    assert execute(case,services,inspection_count)==physical
            assert summarize(data["rows"])==data["summary"]
    assert not split_fps[0]&split_fps[1]
    assert not set(r["train_fingerprints"])&(split_fps[0]|split_fps[1])
    assert set(selection["development_fingerprints"])==split_fps[0]
    return dict(seed=r["seed"],source=r["source"],report_sha256=digest(folder/"report.json"),
                selected=selection["selected"],evaluation={k:v["summary"] for k,v in r["evaluation"].items()})


if __name__=="__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("folders",nargs=3)
    ap.add_argument("--output",required=True)
    a = ap.parse_args()
    torch.set_num_threads(1)
    results = [audit(f) for f in a.folders]
    assert {r["seed"] for r in results}=={4801,4802,4803}
    assert len({r["source"] for r in results})==1
    Path(a.output).write_text(json.dumps(dict(study="DEV-048A",status="CANDIDATE_AND_PHYSICAL_REPLAY_PASSED_DEVELOPMENT_ONLY",
        per_seed=results,confirmatory_claim=False,gepa_llm_reflection_tested=False),indent=2)+"\n")
    print("DEV-048A all-candidate replay and physical verification passed for all three seeds")
