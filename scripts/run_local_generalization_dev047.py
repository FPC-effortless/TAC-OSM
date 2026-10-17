"""DEV-047A variable-ID/size/queue generalization with public event-queue control."""
from __future__ import annotations
import argparse
from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/"src"),str(ROOT/"scripts")]
import torch
from tac_osm.development.local_memory_variants import build_variant
from tac_osm.development.scalable_maintenance import make_pair, catalog_tensors, public_reference, execute
from tac_osm.development.public_state_training import prediction_from_state, outcome_tensor, objective
from tac_osm.development.frozen_action_heads import predictive_metrics
from run_frozen_policy_dev045 import digest
from run_reservoir_scaling_dev044 import THRESHOLDS

PROTOCOL = ROOT/"research/dev047/protocol.json"
VARIANTS = ("multiscale_gru","fast_gru","selective_diagonal","temporal_order")


def training_cases(seed,step,smoke=False):
    roles = ("retention","acquisition","pending_cancel","reissued_cancel")
    deadlines = ("before","at","after")
    return make_pair(seed*1000+step,32 if smoke else (64,128)[step%2],
        (32,128,512)[(step//4)%3],roles[step%4],deadlines[(step//12)%3],domain="dev047a-fit")


def cohort(seed,split,smoke=False):
    out = []
    for n in ((8,) if smoke or split=="calibration" else (8,16,32)):
        for h in ((32,) if smoke else ((128,) if split=="calibration" else (128,512))):
            for multiple in ((False,) if smoke or split=="calibration" else (False,True)):
                for role in (("reissued_cancel",) if smoke else ("pending_cancel","reissued_cancel")):
                    for delay in ((32,) if smoke else (32,512)):
                        for deadline in (("at",) if smoke else ("before","at","after")):
                            cases = make_pair(seed*1000,h,delay,role,deadline,entities=n,
                                multiple_pending=multiple,unseen_ids=split!="calibration",domain="dev047a-"+split)
                            condition = dict(entities=n,history=h,multiple_pending=multiple,role=role,delay=delay,deadline=deadline)
                            out.append((cases,condition))
    return out


@torch.no_grad()
def predict(model,cases,admit=None):
    state = model.encode(*catalog_tensors([c.observations for c in cases],[c.entity_ids for c in cases]),admit=admit)
    return prediction_from_state(model,state,torch.tensor([c.horizon for c in cases],dtype=torch.float32))


def physical_rows(cases,probabilities,threshold):
    rows = []
    for case,p in zip(cases,probabilities.tolist()):
        services = tuple(e for e,q in enumerate(p) if q[0]-q[1]>threshold)
        reference = public_reference(case.observations,case.entity_ids,case.tick,case.horizon)
        assert reference==case.wait_targets
        last = [0]*len(case.entity_ids)
        index = {v:i for i,v in enumerate(case.entity_ids)}
        for o in case.observations:
            if o.kind in ("inspect","change"):
                last[index[o.entity]] = o.value
        rows.append(dict(fingerprint=case.fingerprint,probabilities=p,
            outcomes=list(map(list,zip(case.wait_targets,case.service_targets))),
            physical=execute(case,services),
            public_reference=execute(case,tuple(e for e,v in enumerate(reference) if v)),
            last_value=execute(case,tuple(e for e,v in enumerate(last) if v))))
    return rows


def totals(rows,key="physical"):
    total = {k:sum(r[key][k] for r in rows) for k in ("failures","unnecessary_services","inspections","total_cost")}
    total["services"] = sum(len(r[key]["services"]) for r in rows)
    total["cost_per_entity"] = total["total_cost"]/sum(len(r["probabilities"]) for r in rows)
    return total


def run(seed,output,smoke=False):
    if seed not in (4701,4702,4703):
        raise ValueError("unregistered seed")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    if subprocess.check_output(["git","status","--porcelain"],cwd=ROOT,text=True).strip():
        raise ValueError("freeze source before training")
    source = subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()
    output = Path(output)
    output.mkdir(parents=True,exist_ok=False)
    report = dict(study="DEV-047A",status="DEVELOPMENT_NOT_CONFIRMATORY",seed=seed,source=source,
                  protocol_sha256=digest(PROTOCOL),smoke=smoke,variants={})
    calibration = cohort(seed,"calibration",smoke)
    evaluation = cohort(seed,"evaluation",smoke)
    cal_fps = {c.fingerprint for pair,_ in calibration for c in pair}
    eval_fps = {c.fingerprint for pair,_ in evaluation for c in pair}
    assert not cal_fps&eval_fps
    for variant in VARIANTS:
        model = build_variant(variant,seed)
        opt = torch.optim.AdamW(model.parameters(),lr=.003)
        fingerprints,curve = [],[]
        start = time.monotonic()
        for step in range(2 if smoke else 72):
            cases = training_cases(seed,step,smoke)
            fingerprints.extend(c.fingerprint for c in cases)
            opt.zero_grad(set_to_none=True)
            # predict's no_grad decorator is for evaluation; train explicitly.
            state = model.encode(*catalog_tensors([c.observations for c in cases],[c.entity_ids for c in cases]))
            p = prediction_from_state(model,state,torch.tensor([c.horizon for c in cases],dtype=torch.float32))
            loss = objective(p,outcome_tensor(cases),"joint")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
            opt.step()
            if step%24==0 or step==(1 if smoke else 71):
                curve.append(dict(step=step+1,loss=float(loss.detach())))
        train_seconds = time.monotonic()-start
        assert not set(fingerprints)&cal_fps and not set(fingerprints)&eval_fps
        model.eval()
        torch.save(model.state_dict(),output/(variant+".pt"))
        # Selection completes before evaluation prediction or retention probes.
        cal = [(cases,condition,predict(model,cases)) for cases,condition in calibration]
        scores = []
        for threshold in THRESHOLDS:
            total = sum(totals(physical_rows(cases,p,threshold))["total_cost"] for cases,_,p in cal)
            scores.append((total,abs(threshold-.25),threshold))
        threshold = min(scores)[-1]
        results = []
        for cases,condition in evaluation:
            n = condition["entities"]
            scaled = build_variant(variant,seed,entities=n)
            scaled.load_state_dict(model.state_dict())
            scaled.eval()
            start = time.monotonic()
            p = predict(scaled,cases)
            elapsed = time.monotonic()-start
            # A true slot permutation changes addresses and outputs together.
            permutation = list(reversed(range(n)))
            renamed = [replace(c,entity_ids=tuple(c.entity_ids[i] for i in permutation)) for c in cases]
            pp = predict(scaled,renamed)
            assert torch.equal(pp,p[:,permutation])
            rows = physical_rows(cases,p,threshold)
            mask = torch.ones(len(cases),len(cases[0].observations),dtype=torch.bool)
            for b,c in enumerate(cases):
                earliest = next(i for i,o in enumerate(c.observations) if o.entity==c.entity_ids[c.target_entity])
                mask[b,earliest] = False
            removed = predict(scaled,cases,mask)
            removed_cost = totals(physical_rows(cases,removed,threshold))["total_cost"]
            results.append(dict(condition=condition,rows=rows,summary=totals(rows),
                reference=totals(rows,"public_reference"),last_value=totals(rows,"last_value"),
                metrics=predictive_metrics(p,outcome_tensor(cases)),inference_seconds=elapsed,
                state_scalars=n*25,permutation_invariance=True,
                retention=dict(removed_earliest_target_event=True,
                    delta_future_loss=float(objective(removed,outcome_tensor(cases),"forecast")-
                                            objective(p,outcome_tensor(cases),"forecast")),
                    delta_total_cost=removed_cost-totals(rows)["total_cost"])))
        report["variants"][variant] = dict(checkpoint_sha256=digest(output/(variant+".pt")),
            threshold=threshold,calibration=[dict(condition=cond,rows=physical_rows(cs,p,threshold)) for cs,cond,p in cal],
            train_fingerprints=fingerprints,training_seconds=train_seconds,curve=curve,
            parameters=sum(p.numel() for p in model.parameters()),evaluation=results)
        print(json.dumps(dict(seed=seed,variant=variant,
            cost=totals([row for r in results for row in r["rows"]])["cost_per_entity"],training_seconds=train_seconds)),flush=True)
    (output/"report.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")


if __name__=="__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed",required=True,type=int)
    ap.add_argument("--output",required=True)
    ap.add_argument("--smoke",action="store_true")
    a = ap.parse_args()
    run(a.seed,a.output,a.smoke)
