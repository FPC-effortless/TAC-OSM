"""DEV-048A sparse prediction, measured fallbacks and paid verified consequences."""
from __future__ import annotations
import argparse
import copy
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/"src"),str(ROOT/"scripts")]
import torch
from tac_osm.development.unified_state_prototype import UnifiedPredictiveStatePrototype
from tac_osm.development.scalable_maintenance import make_pair, catalog_tensors, public_reference, execute
from tac_osm.development.public_state_training import public_tensors, outcome_tensor
from tac_osm.development.verified_routing import sparse_plan, admissible_services, commit_public_receipts
from tac_osm.development.delayed_maintenance import Action
from tac_osm.development.decision_cost_policy import select_actions, Decision
from tac_osm.development.noisy_memory import parameter_hash
from run_frozen_policy_dev045 import digest

PROTOCOL = ROOT/"research/dev048/protocol.json"
CANDIDATES = ("exact_history","route2_exact","route4_exact","route8_exact",
              "route2_inspect","route4_inspect","route2_review","route4_review")


def cohort(seed,split,smoke=False):
    for n in ((8,) if smoke or split=="development" else (8,16,32)):
        for h in ((32,) if smoke else ((128,) if split=="development" else (128,512))):
            for role in (("reissued_cancel",) if smoke else ("pending_cancel","reissued_cancel")):
                for delay in ((32,) if smoke else (32,512)):
                    for deadline in (("at",) if smoke else (("before","at","after") if split=="development" else ("at","after"))):
                        pair = make_pair(seed*1000,h,delay,role,deadline,entities=n,
                            domain="dev048a-"+split,multiple_pending=split!="development",unseen_ids=split!="development")
                        yield pair,dict(entities=n,history=h,role=role,delay=delay,deadline=deadline)


def model_for(state,entities):
    model = UnifiedPredictiveStatePrototype(entities=entities,dim=8,budget=entities)
    model.load_state_dict(state)
    return model.eval()


@torch.no_grad()
def run_policy(state,cases,configuration):
    batch,n = len(cases),len(cases[0].entity_ids)
    times = dict(encoding=0.,routing_prediction=0.,fallback=0.,execution_verification=0.)
    counters = dict(routing_candidates=0,osm_positions=0,history_events_scanned=0,fallback_inspections=0,verification_inspections=0)
    p = None
    latent = None
    model = None
    selected = None
    if configuration=="exact_history":
        start = time.monotonic()
        actions = torch.tensor([public_reference(c.observations,c.entity_ids,c.tick,c.horizon) for c in cases],dtype=torch.long)
        times["fallback"] = time.monotonic()-start
        counters["history_events_scanned"] = sum(len(c.observations) for c in cases)
    else:
        budget = int(configuration.split("_")[0].removeprefix("route"))
        # The 8-slot arm keeps a fixed budget at 16/32 entities too.
        fallback = configuration.split("_")[1]
        model = model_for(state,n)
        start = time.monotonic()
        latent = model.encode(*catalog_tensors([c.observations for c in cases],[c.entity_ids for c in cases]))
        times["encoding"] = time.monotonic()-start
        horizons = torch.tensor([c.horizon for c in cases],dtype=torch.float32)
        start = time.monotonic()
        plan = sparse_plan(model,latent,horizons,min(budget,n))
        times["routing_prediction"] = time.monotonic()-start
        actions,selected = plan.actions.clone(),plan.selected
        counters["routing_candidates"] = plan.routing_candidates
        counters["osm_positions"] = plan.predicted_entities
        p = torch.full((batch,n,2),float("nan"))
        p.scatter_(1,selected[...,None].expand(-1,-1,2),plan.probabilities)
        unresolved = actions==Decision.HUMAN_REVIEW
        start = time.monotonic()
        if fallback=="exact" and unresolved.any():
            reference = torch.tensor([public_reference(c.observations,c.entity_ids,c.tick,c.horizon) for c in cases],dtype=torch.long)
            actions = torch.where(unresolved,reference,actions)
            counters["history_events_scanned"] += sum(len(c.observations) for c in cases)
        elif fallback=="inspect" and unresolved.any():
            worlds = [copy.deepcopy(c.world) for c in cases]
            for e in range(n):
                accepted = unresolved[:,e]
                if not accepted.any():
                    continue
                receipts = [(world.act(Action("inspect",e)),) for world in worlds]
                features,ids,ticks = public_tensors(receipts)
                latent = commit_public_receipts(model,latent,features[:,0],ids[:,0],ticks[:,0],accepted)
                counters["fallback_inspections"] += int(accepted.sum())
            slots = model.memory.read(latent)
            for b in range(batch):
                ids = torch.nonzero(unresolved[b]).flatten()
                if len(ids):
                    pw,ps = model.world_predictor(slots[b:b+1,ids],horizons[b:b+1])
                    actions[b,ids] = select_actions(pw,ps).action[0]
                    p[b,ids] = torch.stack((pw,ps),-1)[0]
                    counters["osm_positions"] += len(ids)
        times["fallback"] = time.monotonic()-start
    coverage = float((actions!=Decision.HUMAN_REVIEW).float().mean())
    if coverage<1:
        return dict(configuration=configuration,fingerprints=[c.fingerprint for c in cases],
                    actions=actions.tolist(),coverage=coverage,physical=None,counters=counters,times=times,
                    note="Unresolved review cost is unknown; no complete-system cost comparison.")
    services = admissible_services(actions)
    start = time.monotonic()
    outcomes = []
    worlds = [copy.deepcopy(c.world) for c in cases]
    for e in range(n):
        accepted = actions[:,e]==Decision.SERVICE
        receipts = [(world.act(Action("service" if bool(accepted[b]) else "wait",e,0,c.horizon)),)
                    for b,(world,c) in enumerate(zip(worlds,cases))]
        if model is not None:
            features,ids,ticks = public_tensors(receipts)
            latent = commit_public_receipts(model,latent,features[:,0],ids[:,0],ticks[:,0],accepted)
    for world,c in zip(worlds,cases):
        for _ in range(c.horizon):
            world.advance()
    verification = []
    for e in range(n):
        receipts = [(world.act(Action("inspect",e)),) for world in worlds]
        verification.append([r[0].value for r in receipts])
        counters["verification_inspections"] += batch
        if model is not None:
            features,ids,ticks = public_tensors(receipts)
            latent = commit_public_receipts(model,latent,features[:,0],ids[:,0],ticks[:,0],torch.ones(batch,dtype=torch.bool))
    for b,(c,chosen) in enumerate(zip(cases,services)):
        # Charge individual fallback sensing and all post-horizon verification.
        before = n-len(selected[b]) if configuration.endswith("_inspect") else 0
        physical = execute(c,chosen,before+n)
        observed = [verification[e][b] for e in range(n)]
        assert physical["outcomes"]==observed
        outcomes.append(physical)
    times["execution_verification"] = time.monotonic()-start
    return dict(configuration=configuration,fingerprints=[c.fingerprint for c in cases],actions=actions.tolist(),
                coverage=coverage,physical=outcomes,counters=counters,times=times,verified=True)


def summarize(rows):
    complete = all(r["physical"] is not None for r in rows)
    counters = {k:sum(r["counters"][k] for r in rows) for k in rows[0]["counters"]}
    times = {k:sum(r["times"][k] for r in rows) for k in rows[0]["times"]}
    result = dict(complete=complete,coverage=sum(r["coverage"] for r in rows)/len(rows),counters=counters,times=times)
    if complete:
        physical = [p for r in rows for p in r["physical"]]
        result.update({k:sum(p[k] for p in physical) for k in ("failures","unnecessary_services","inspections","total_cost")})
        result["cost_per_entity"] = result["total_cost"]/sum(len(p["outcomes"]) for p in physical)
    return result


def choose(development):
    complete = {k:v for k,v in development.items() if v["summary"]["complete"]}
    selected = min(complete,key=lambda k:(complete[k]["summary"]["failures"],complete[k]["summary"]["cost_per_entity"],
                                         complete[k]["summary"]["counters"]["osm_positions"],k))
    pareto = []
    for name,data in complete.items():
        a = data["summary"]
        av = (a["failures"],a["cost_per_entity"],sum(a["times"].values()))
        dominated = False
        for other,d in complete.items():
            b = d["summary"]
            bv = (b["failures"],b["cost_per_entity"],sum(b["times"].values()))
            if other!=name and all(x<=y for x,y in zip(bv,av)) and any(x<y for x,y in zip(bv,av)):
                dominated = True
        if not dominated:
            pareto.append(name)
    return selected,pareto


def run(parent_root,parent_audit,seed,output,smoke=False):
    if seed not in (4801,4802,4803):
        raise ValueError("unregistered seed")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    if subprocess.check_output(["git","status","--porcelain"],cwd=ROOT,text=True).strip():
        raise ValueError("freeze source before running")
    source = subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()
    audit = json.loads(Path(parent_audit).read_text())
    assert audit["study"]=="DEV-046B" and audit["status"]=="CHECKPOINT_AND_PHYSICAL_REPLAY_PASSED_DEVELOPMENT_ONLY"
    assert {r["seed"] for r in audit["per_seed"]}=={4601,4602,4603}
    parent = Path(parent_root)/str(seed-200)
    audited = next(r for r in audit["per_seed"] if r["seed"]==seed-200)
    assert audited["report_sha256"]==digest(parent/"report.json")
    report046 = json.loads((parent/"report.json").read_text())
    ck = parent/"joint_trainable.pt"
    assert digest(ck)==report046["arms"]["joint_trainable"]["checkpoint_sha256"]
    model = model_for(torch.load(ck,weights_only=True,map_location="cpu"),8)
    model.requires_grad_(False)
    frozen_hash = parameter_hash(model.memory)+parameter_hash(model.observation)+parameter_hash(model.world_predictor)
    model.relevance.requires_grad_(True)
    opt = torch.optim.AdamW(model.relevance.parameters(),lr=.003)
    train_fps = []
    for step in range(2 if smoke else 72):
        pair = make_pair(seed*1000+step,32 if smoke else 128,(32,128,512)[step%3],
                         ("pending_cancel","reissued_cancel")[step%2],("before","at","after")[(step//2)%3],domain="dev048a-fit")
        train_fps.extend(c.fingerprint for c in pair)
        with torch.no_grad():
            latent = model.encode(*catalog_tensors([c.observations for c in pair],[c.entity_ids for c in pair]))
            slots = model.memory.read(latent)
        label = (4*outcome_tensor(pair)[...,0]-1-4*outcome_tensor(pair)[...,1]).clamp_min(0)
        loss = (model.relevance.score(slots).squeeze(-1)-label).square().mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    assert frozen_hash==parameter_hash(model.memory)+parameter_hash(model.observation)+parameter_hash(model.world_predictor)
    output = Path(output)
    output.mkdir(parents=True,exist_ok=False)
    torch.save(model.state_dict(),output/"router-model.pt")
    state = model.state_dict()
    development = {}
    for candidate in CANDIDATES:
        rows = [dict(condition=condition,**run_policy(state,cases,candidate)) for cases,condition in cohort(seed,"development",smoke)]
        development[candidate] = dict(rows=rows,summary=summarize(rows))
    selected,pareto = choose(development)
    selection = dict(selected=selected,pareto=pareto,checkpoint_sha256=digest(output/"router-model.pt"),
                     development_fingerprints=[c.fingerprint for cs,_ in cohort(seed,"development",smoke) for c in cs])
    (output/"selection-lock.json").write_text(json.dumps(selection,indent=2)+"\n")
    evaluation = {}
    for candidate in CANDIDATES:
        rows = [dict(condition=condition,**run_policy(state,cases,candidate)) for cases,condition in cohort(seed,"evaluation",smoke)]
        evaluation[candidate] = dict(rows=rows,summary=summarize(rows))
    eval_fps = {c.fingerprint for cs,_ in cohort(seed,"evaluation",smoke) for c in cs}
    assert not set(selection["development_fingerprints"])&eval_fps and not set(train_fps)&eval_fps
    report = dict(study="DEV-048A",status="DEVELOPMENT_NOT_CONFIRMATORY",seed=seed,source=source,smoke=smoke,
                  protocol_sha256=digest(PROTOCOL),parent_audit_sha256=digest(parent_audit),
                  parent_checkpoint_sha256=digest(ck),checkpoint_sha256=digest(output/"router-model.pt"),
                  selection_lock_sha256=digest(output/"selection-lock.json"),train_fingerprints=train_fps,
                  development=development,selection=selection,evaluation=evaluation)
    (output/"report.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    print(json.dumps(dict(seed=seed,selected=selected,evaluation={k:v["summary"] for k,v in evaluation.items()})),flush=True)


if __name__=="__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--parent-root",required=True)
    ap.add_argument("--parent-audit",required=True)
    ap.add_argument("--seed",required=True,type=int)
    ap.add_argument("--output",required=True)
    ap.add_argument("--smoke",action="store_true")
    a = ap.parse_args()
    run(a.parent_root,a.parent_audit,a.seed,a.output,a.smoke)
