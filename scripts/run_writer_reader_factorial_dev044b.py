"""DEV-044B0: independently train/freeze internal writer and query-read parameters."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import random
import subprocess
import sys
import time
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/"src"),str(ROOT/"scripts")]
import torch
from torch.nn import functional as F
from tac_osm.development.bounded_mtsk import build_bounded
from tac_osm.development.writer_reader_factorial import ARMS,configure,group_hash,gradient_present
from tac_osm.development.schema_maintenance import make_pair,ROLES,DEADLINES
from tac_osm.development.delayed_maintenance import Action
from tac_osm.development.finite_clipping import clip_finite_cpu_gradients_
from run_learned_schema_dev import public_stream_tensors,public_queries,labels
from run_frozen_policy_dev045 import cohort,evaluate,rows_for,threshold_for,digest

PROTOCOL = ROOT/"research/dev044b/protocol.json"


def build(arm,seed):
    return configure(build_bounded(seed+40000),arm)


def training_cases(seed,step,smoke=False):
    return make_pair(seed*100000+step,32 if smoke else 128,(32,128,512)[(step//4)%3],
                     ROLES[step%4],DEADLINES[(step//12)%3],domain="dev044b0-fit")


def study_cohort(seed,split,smoke=False):
    return cohort(seed+200000,"dev044b0-"+split,1 if smoke or split=="calibration" else 2,smoke)


@torch.no_grad()
def predict(model,cases):
    result = []
    for i in range(0,len(cases),16):
        batch = cases[i:i+16]
        result.append(model.decode(model.encode(*public_stream_tensors([c.observations for c in batch])),
            public_queries([c.tick for c in batch],[c.horizon for c in batch])).softmax(-1)[...,:2,1])
    return torch.cat(result)


def run(seed,output,smoke=False):
    assert seed in (4411,4412,4413)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    assert not subprocess.check_output(["git","status","--porcelain"],cwd=ROOT,text=True).strip(),"freeze source"
    source = subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()
    output = Path(output)
    output.mkdir(parents=True,exist_ok=False)
    cohorts = {s:study_cohort(seed,s,smoke) for s in ("calibration","evaluation")}
    r = dict(study="DEV-044B0",status="DEVELOPMENT_NOT_CONFIRMATORY",seed=seed,smoke=smoke,
             source=source,protocol_sha256=digest(PROTOCOL),arms={})
    initial = build(ARMS[0],seed).state_dict()
    for arm in ARMS:
        model = build(arm,seed)
        assert all(torch.equal(v,model.state_dict()[k]) for k,v in initial.items())
        before = {name:group_hash(model,reader) for name,reader in (("writer",False),("reader",True))}
        seen = dict(writer=False,reader=False)
        opt = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),lr=.002,weight_decay=.0001)
        fingerprints,curve = [],[]
        start = time.monotonic()
        for step in range(2 if smoke else 108):
            cases = training_cases(seed,step,smoke)
            fingerprints.extend(c.fingerprint for c in cases)
            state = model.encode(*public_stream_tensors([c.observations for c in cases]))
            q = public_queries([c.tick for c in cases],[c.horizon for c in cases])
            y = labels(cases)
            receipts = [(c.world.act(Action("inspect",random.Random(f"dev044b0:{seed}:{step}:{b}").randrange(8))),)
                        for b,c in enumerate(cases)]
            receipt = public_stream_tensors(receipts)
            pre = model.decode(state,q)
            post = model.decode(model.update(state,*(x[:,0] for x in receipt)),q)
            loss = (F.cross_entropy(pre.reshape(-1,2),y.reshape(-1))+F.cross_entropy(post.reshape(-1,2),y.reshape(-1)))/2
            opt.zero_grad(set_to_none=True)
            loss.backward()
            seen["writer"] |= gradient_present(model,False)
            seen["reader"] |= gradient_present(model,True)
            clip_finite_cpu_gradients_(model.parameters(),1.)
            opt.step()
            if step%36==0 or step==(1 if smoke else 107):
                curve.append(dict(step=step+1,loss=float(loss.detach())))
        after = {name:group_hash(model,reader) for name,reader in (("writer",False),("reader",True))}
        expected = dict(writer=arm.startswith("writer_train_"),reader=arm.endswith("_reader_train"))
        assert seen==expected
        assert all((before[k]!=after[k])==v for k,v in expected.items())
        train_seconds = time.monotonic()-start
        model.eval()
        torch.save(model.state_dict(),output/(arm+".pt"))
        cal = predict(model,cohorts["calibration"][0])
        threshold = threshold_for(rows_for(*cohorts["calibration"],cal))
        ev = predict(model,cohorts["evaluation"][0])
        r["arms"][arm] = dict(before_hashes=before,after_hashes=after,gradient_present=seen,
            train_fingerprints=fingerprints,curve=curve,training_seconds=train_seconds,
            checkpoint_sha256=digest(output/(arm+".pt")),threshold=threshold,
            calibration_rows=rows_for(*cohorts["calibration"],cal),evaluation=evaluate(*cohorts["evaluation"],ev,threshold))
        print(json.dumps(dict(seed=seed,arm=arm,physical=r["arms"][arm]["evaluation"]["physical"])),flush=True)
    (output/"report.json").write_text(json.dumps(r,indent=2,allow_nan=False)+"\n")


if __name__=="__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed",required=True,type=int)
    ap.add_argument("--output",required=True)
    ap.add_argument("--smoke",action="store_true")
    a = ap.parse_args()
    run(a.seed,a.output,a.smoke)
