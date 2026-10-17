"""DEV-045B frozen public readout versus probability-only action interfaces."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
import torch
from tac_osm.development.frozen_action_heads import (
    FrozenActionHead, action_head_loss, frozen_features, predictive_metrics,
)
from tac_osm.development.noisy_memory import parameter_hash
from tac_osm.development.schema_maintenance import (
    make_pair, ROLES, DEADLINES, execute, exact_public_targets,
)
from run_reservoir_scaling_dev044 import build, THRESHOLDS, physical, last_keys
from run_learned_schema_dev import public_stream_tensors, public_queries

PROTOCOL = ROOT / "research/dev045/study-protocol.json"
HEADS = ("forecast", "decision", "contrastive", "joint", "probability_joint")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def cohort(seed, split, repetitions, smoke=False):
    cases, conditions = [], []
    for delay in ((32,) if smoke else (32, 128, 512)):
        for role in (("reissued_cancel",) if smoke else ROLES):
            for deadline in (("at",) if smoke else DEADLINES):
                for rep in range(repetitions):
                    pair = make_pair(seed * 1000 + rep, 32 if smoke else 512,
                                     delay, role, deadline, domain="dev045b-" + split)
                    for partner, case in enumerate(pair):
                        cases.append(case)
                        conditions.append(dict(fingerprint=case.fingerprint, delay=delay,
                                               role=role, deadline=deadline, repetition=rep,
                                               partner=partner, target_entity=case.target_entity))
    return cases, conditions


def gather(model, cases):
    features, original = [], []
    started = time.monotonic()
    for offset in range(0, len(cases), 16):
        batch = cases[offset:offset+16]
        f, p = frozen_features(model, public_stream_tensors([c.observations for c in batch]),
                               public_queries([c.tick for c in batch], [c.horizon for c in batch]))
        features.append(f)
        original.append(p)
    return torch.cat(features), torch.cat(original), time.monotonic() - started


def targets(cases):
    return torch.tensor([list(zip(c.wait_targets, c.service_targets)) for c in cases], dtype=torch.float32)


def fit(features, y, objective, seed, smoke):
    torch.manual_seed(seed)
    head = FrozenActionHead(features.shape[-1])
    opt = torch.optim.AdamW(head.parameters(), lr=.003)
    x, y = features.flatten(0, 1), y.flatten(0, 1)
    curve = []
    started = time.monotonic()
    for step in range(2 if smoke else 240):
        idx = torch.randperm(len(x))[:128]
        opt.zero_grad(set_to_none=True)
        loss = action_head_loss(head(x[idx]), y[idx], objective)
        if not torch.isfinite(loss):
            raise FloatingPointError("nonfinite action-head loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), 1.)
        opt.step()
        if step % 60 == 0 or step == (1 if smoke else 239):
            curve.append(dict(step=step+1, loss=float(loss.detach())))
    return head.eval(), dict(seconds=time.monotonic()-started, curve=curve,
                             trainable_parameters=sum(p.numel() for p in head.parameters()))


def rows_for(cases, conditions, probabilities):
    return [dict(**cond, probabilities=p, labels=list(map(list, zip(c.wait_targets, c.service_targets))))
            for c, cond, p in zip(cases, conditions, probabilities.tolist())]


def threshold_for(rows):
    return min(THRESHOLDS, key=lambda t: (physical(rows, t)["total_cost"], abs(t-.25), t))


def evaluate(cases, conditions, probabilities, threshold):
    rows = rows_for(cases, conditions, probabilities)
    costs = physical(rows, threshold)
    oracle_total, agreements = 0., 0
    for row, case in zip(rows, cases):
        services = tuple(e for e, p in enumerate(row["probabilities"]) if p[0]-p[1] > threshold)
        replay = execute(case.world, case.horizon, services, 0)
        oracle = tuple(e for e in range(8) if 1+4*case.service_targets[e] < 4*case.wait_targets[e])
        oracle_total += execute(case.world, case.horizon, oracle, 0)["total_cost"]
        agreements += sum((e in services) == (e in oracle) for e in range(8))
        row["services"] = list(services)
        row["physical"] = replay
    assert costs["total_cost"] == sum(r["physical"]["total_cost"] for r in rows)
    margins = []
    for i in range(0, len(rows), 2):
        a, b = rows[i:i+2]
        e = a["target_entity"]
        if a["deadline"] != "before":
            dy = a["labels"][e][0] - b["labels"][e][0]
            margins.append((a["probabilities"][e][0] - b["probabilities"][e][0]) * dy)
    return dict(rows=rows, physical=costs,
                metrics=dict(**predictive_metrics(probabilities, targets(cases)),
                             action_regret_per_entity=(costs["total_cost"]-oracle_total)/(8*len(cases)),
                             teacher_agreement=agreements/(8*len(cases)),
                             signed_pair_margins=margins,
                             pair_ranking=(sum(v>0 for v in margins)+.5*sum(v==0 for v in margins))/len(margins)))


def run(parent_root, audit_path, seed, output, smoke=False):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    source = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip():
        raise ValueError("freeze all source before running")
    if seed not in (4501, 4502, 4503):
        raise ValueError("unregistered seed")
    parent_seed = seed-100
    audit = json.loads(Path(audit_path).read_text())
    if audit["status"] != "AUDITED_DEVELOPMENT_NOT_CONFIRMATORY" or {x["seed"] for x in audit["per_seed"]} != {4401, 4402, 4403}:
        raise ValueError("DEV-044A complete audit prerequisite not met")
    parent = Path(parent_root) / str(parent_seed)
    report044 = json.loads((parent/"report.json").read_text())
    parent_audit = next(x for x in audit["per_seed"] if x["seed"] == parent_seed)
    if parent_audit["selected"] != report044["selected"]:
        raise ValueError("audited selection mismatch")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    cohorts = {split: cohort(seed, split, 1 if smoke or split != "evaluation" else 2, smoke)
               for split in ("fit", "calibration", "evaluation")}
    fps = [{c.fingerprint for c in cs} for cs, _ in cohorts.values()]
    assert all(not a & b for i, a in enumerate(fps) for b in fps[i+1:])
    report = dict(study="DEV-045B", source=source, seed=seed, smoke=smoke,
                  status="DEVELOPMENT_NOT_CONFIRMATORY", protocol_sha256=digest(PROTOCOL),
                  parent_audit_sha256=digest(audit_path), memories={})
    for memory in ("selected_trainable", "selected_frozen", "schema41"):
        arm = report044["selected"][memory.removeprefix("selected_")] if memory != "schema41" else memory
        ck = parent/(arm+".pt")
        assert digest(ck) == report044["arms"][arm]["checkpoint_sha256"]
        model = build(arm, parent_seed)
        model.load_state_dict(torch.load(ck, weights_only=True, map_location="cpu")["model"])
        model.eval().requires_grad_(False)
        frozen_hash = parameter_hash(model)
        data = {split: gather(model, cases) for split, (cases, _) in cohorts.items()}
        torch.save({s: {"features": f, "original": p, "targets": targets(cohorts[s][0])}
                    for s, (f, p, _) in data.items()}, output/(memory+"-public-features.pt"))
        result = dict(parent_arm=arm, parent_checkpoint_sha256=digest(ck), frozen_model_hash=frozen_hash,
                      feature_file_sha256=digest(output/(memory+"-public-features.pt")),
                      inference_seconds={s: v[2] for s,v in data.items()}, heads={})
        for name in HEADS:
            index = 1 if name == "probability_joint" else 0
            head, info = fit(data["fit"][index], targets(cohorts["fit"][0]), name, seed, smoke)
            with torch.no_grad():
                cal = head(data["calibration"][index]).sigmoid()
                ev = head(data["evaluation"][index]).sigmoid()
            cal_rows = rows_for(*cohorts["calibration"], cal)
            threshold = threshold_for(cal_rows)
            torch.save(head.state_dict(), output/(memory+"-"+name+".pt"))
            result["heads"][name] = dict(training=info, threshold=threshold,
                checkpoint_sha256=digest(output/(memory+"-"+name+".pt")), calibration_rows=cal_rows,
                evaluation=evaluate(*cohorts["evaluation"], ev, threshold))
        threshold = threshold_for(rows_for(*cohorts["calibration"], data["calibration"][1]))
        result["heads"]["original_tuned_threshold"] = dict(threshold=threshold,
            evaluation=evaluate(*cohorts["evaluation"], data["evaluation"][1], threshold))
        assert parameter_hash(model) == frozen_hash
        report["memories"][memory] = result
        print(json.dumps(dict(seed=seed, memory=memory,
              costs={n:v["evaluation"]["physical"]["cost_per_entity"] for n,v in result["heads"].items()})), flush=True)
    table = {}
    for c in cohorts["fit"][0]:
        for key, y in zip(last_keys(c), zip(c.wait_targets, c.service_targets)):
            total, count = table.get(key, ([0.,0.], 0))
            table[key] = ([a+b for a,b in zip(total,y)], count+1)
    def table_probs(cases):
        return torch.tensor([[[(table[k][0][i]/table[k][1] if k in table else .5) for i in range(2)]
                              for k in last_keys(c)] for c in cases])
    threshold = threshold_for(rows_for(*cohorts["calibration"], table_probs(cohorts["calibration"][0])))
    report["last_event_table"] = dict(threshold=threshold,
        evaluation=evaluate(*cohorts["evaluation"], table_probs(cohorts["evaluation"][0]), threshold))
    exact = torch.tensor([list(zip(exact_public_targets(c.observations,c.tick,c.horizon), [0]*8))
                          for c in cohorts["evaluation"][0]], dtype=torch.float32)
    report["exact_public_history"] = evaluate(*cohorts["evaluation"], exact, .25)
    report["split_fingerprints"] = {s:[c.fingerprint for c in cs] for s,(cs,_) in cohorts.items()}
    (output/"report.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--parent-root", required=True)
    ap.add_argument("--parent-audit", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    run(a.parent_root, a.parent_audit, a.seed, a.output, a.smoke)
