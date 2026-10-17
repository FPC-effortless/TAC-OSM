"""DEV-046B joint/frozen and simulator-teaching ablations, exact physical audit data."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/"src"),str(ROOT/"scripts")]
import torch
from tac_osm.development.unified_state_prototype import UnifiedPredictiveStatePrototype
from tac_osm.development.public_state_training import (
    public_tensors, prediction_from_state, objective, outcome_tensor, student_rollout,
)
from tac_osm.development.schema_maintenance import make_pair, ROLES, DEADLINES, exact_public_targets
from tac_osm.development.noisy_memory import parameter_hash
from run_frozen_policy_dev045 import cohort, evaluate, rows_for, threshold_for, digest

PROTOCOL = ROOT/"research/dev046/study-protocol.json"
ARMS = tuple(f"{objective}_{condition}" for condition in ("trainable","frozen")
             for objective in ("forecast","decision","joint","teacher"))


def build(arm, seed):
    if arm not in ARMS:
        raise ValueError("unregistered arm")
    torch.manual_seed(seed)
    model = UnifiedPredictiveStatePrototype(entities=8,dim=8,budget=8)
    if arm.endswith("_frozen"):
        model.observation.requires_grad_(False)
        model.memory.requires_grad_(False)
    return model


def writer_hash(model):
    return hashlib.sha256((parameter_hash(model.observation)+parameter_hash(model.memory)).encode()).hexdigest()


def training_cases(seed, step, smoke):
    return make_pair(seed*100000+step,32 if smoke else (64,128)[step%2],
        (32,128,512)[(step//4)%3],ROLES[step%4],DEADLINES[(step//12)%3],domain="dev046b-fit")


def study_cohort(seed, split, smoke=False):
    # Reuse only the condition grid; namespace differs from DEV-045B.
    return cohort(seed+100000, "dev046b-"+split, 1 if smoke or split=="calibration" else 2,smoke)


@torch.no_grad()
def predict(model, cases):
    out = []
    start = time.monotonic()
    for i in range(0,len(cases),16):
        batch = cases[i:i+16]
        state = model.encode(*public_tensors([c.observations for c in batch]))
        out.append(prediction_from_state(model,state,torch.tensor([c.horizon for c in batch],dtype=torch.float32)))
    return torch.cat(out),time.monotonic()-start


def run(seed, output, smoke=False):
    if seed not in (4601,4602,4603):
        raise ValueError("unregistered seed")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    if subprocess.check_output(["git","status","--porcelain"],cwd=ROOT,text=True).strip():
        raise ValueError("freeze source before training")
    source = subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()
    output = Path(output)
    output.mkdir(parents=True,exist_ok=False)
    cohorts = {s:study_cohort(seed,s,smoke) for s in ("calibration","evaluation")}
    r = dict(study="DEV-046B",status="DEVELOPMENT_NOT_CONFIRMATORY",source=source,
             protocol_sha256=digest(PROTOCOL),seed=seed,smoke=smoke,arms={})
    for arm in ARMS:
        model = build(arm,seed)
        initial_writer = writer_hash(model)
        opt = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),lr=.003)
        curve, fingerprints = [], []
        receipts = dict(services=0,inspections=0)
        start = time.monotonic()
        for step in range(2 if smoke else 72):
            cases = training_cases(seed,step,smoke)
            fingerprints.extend(c.fingerprint for c in cases)
            state = model.encode(*public_tensors([c.observations for c in cases]))
            horizons = torch.tensor([c.horizon for c in cases],dtype=torch.float32)
            probabilities = prediction_from_state(model,state,horizons)
            kind = arm.split("_")[0]
            loss = objective(probabilities,outcome_tensor(cases),"joint" if kind=="teacher" else kind)
            if kind=="teacher":
                next_state,y,counts = student_rollout(model,state,cases,probabilities)
                loss = loss + objective(prediction_from_state(model,next_state,horizons),y,"joint")
                for k,v in counts.items():
                    receipts[k] += v
            if not torch.isfinite(loss):
                raise FloatingPointError("nonfinite joint loss")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
            opt.step()
            if step%24==0 or step==(1 if smoke else 71):
                curve.append(dict(step=step+1,loss=float(loss.detach())))
        train_seconds = time.monotonic()-start
        final_writer = writer_hash(model)
        assert (initial_writer==final_writer) == arm.endswith("_frozen")
        model.eval()
        ck = output/(arm+".pt")
        torch.save(model.state_dict(),ck)
        cal,cal_seconds = predict(model,cohorts["calibration"][0])
        threshold = threshold_for(rows_for(*cohorts["calibration"],cal))
        ev,ev_seconds = predict(model,cohorts["evaluation"][0])
        r["arms"][arm] = dict(checkpoint_sha256=digest(ck),initial_writer_hash=initial_writer,
            final_writer_hash=final_writer,train_fingerprints=fingerprints,curve=curve,
            training_seconds=train_seconds,training_receipts=receipts,
            trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
            state_scalars=8*(3*8+1),inference_seconds=ev_seconds,calibration_seconds=cal_seconds,
            threshold=threshold,calibration_rows=rows_for(*cohorts["calibration"],cal),
            evaluation=evaluate(*cohorts["evaluation"],ev,threshold),
            native_cost_policy=evaluate(*cohorts["evaluation"],ev,.25)["physical"])
        assert not set(fingerprints)&{c.fingerprint for c in cohorts["evaluation"][0]}
        assert not set(fingerprints)&{c.fingerprint for c in cohorts["calibration"][0]}
        print(json.dumps(dict(seed=seed,arm=arm,cost=r["arms"][arm]["evaluation"]["physical"],
                             training_seconds=train_seconds)),flush=True)
    cs,conditions = cohorts["evaluation"]
    exact = torch.tensor([list(zip(exact_public_targets(c.observations,c.tick,c.horizon),[0]*8))
                          for c in cs],dtype=torch.float32)
    r["exact_public_history"] = evaluate(cs,conditions,exact,.25)
    r["split_fingerprints"] = {s:[c.fingerprint for c in cases] for s,(cases,_) in cohorts.items()}
    assert not set(r["split_fingerprints"]["calibration"])&set(r["split_fingerprints"]["evaluation"])
    (output/"report.json").write_text(json.dumps(r,indent=2,allow_nan=False)+"\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed",required=True,type=int)
    ap.add_argument("--output",required=True)
    ap.add_argument("--smoke",action="store_true")
    a = ap.parse_args()
    run(a.seed,a.output,a.smoke)
