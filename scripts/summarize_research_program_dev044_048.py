"""Summarize every audited development arm without selecting evaluation winners."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import statistics


def contrast(rows,reference):
    differences = [a["cost_per_entity"]-b["cost_per_entity"] for a,b in zip(rows,reference)]
    guard = all(a["failures"]<=b["failures"] for a,b in zip(rows,reference))
    return dict(mean_cost_per_entity=statistics.mean(a["cost_per_entity"] for a in rows),
                failures_per_seed=[a["failures"] for a in rows],
                mean_cost_difference=statistics.mean(differences),paired_cost_differences=differences,
                failure_guard_all_seeds=guard,confirmatory_superiority_claim=False)


def summarize(root):
    root = Path(root)
    paths = {s:root/("dev"+s)/("analysis/analysis.json" if s=="044" else "analysis.json") for s in ("044","044b","045","046","047","048")}
    data = {s:json.loads(p.read_text()) for s,p in paths.items()}
    for s,d in data.items():
        assert d["study"].lower().startswith("dev-"+s)
        assert len(d["per_seed"])==3 and len({r["seed"] for r in d["per_seed"]})==3
        assert d["status"] in ("AUDITED_DEVELOPMENT_NOT_CONFIRMATORY",
            "CHECKPOINT_AND_PHYSICAL_REPLAY_PASSED_DEVELOPMENT_ONLY",
            "CANDIDATE_AND_PHYSICAL_REPLAY_PASSED_DEVELOPMENT_ONLY",
            "FACTORIAL_CHECKPOINT_AND_PHYSICAL_REPLAY_PASSED_DEVELOPMENT_ONLY")
    out = dict(status="SIX_AUDITED_DEVELOPMENT_STUDIES_NOT_CONFIRMATORY",training_seeds_per_study=3,
               new_fitted_arms=96,full_architecture_superiority_established=False,studies={},
               audit_sha256={s:hashlib.sha256(p.read_bytes()).hexdigest() for s,p in paths.items()})
    seeds = sorted(data["044"]["per_seed"],key=lambda r:r["seed"])
    out["studies"]["DEV-044A"] = contrast(
        [r["arms"][r["selected"]["trainable"]] for r in seeds],
        [r["arms"][r["selected"]["frozen"]] for r in seeds])
    seeds = sorted(data["044b"]["per_seed"],key=lambda r:r["seed"])
    out["studies"]["DEV-044B0"] = {arm:contrast([r["arms"][arm] for r in seeds],
        [r["arms"]["writer_frozen_reader_"+arm.split("_reader_")[1]] for r in seeds])
        for arm in seeds[0]["arms"]}
    out["studies"]["DEV-044B0"]["writer_by_reader_interaction"] = [
        (r["arms"]["writer_train_reader_train"]["cost_per_entity"]-r["arms"]["writer_frozen_reader_train"]["cost_per_entity"])-
        (r["arms"]["writer_train_reader_frozen"]["cost_per_entity"]-r["arms"]["writer_frozen_reader_frozen"]["cost_per_entity"])
        for r in seeds]
    seeds = sorted(data["045"]["per_seed"],key=lambda r:r["seed"])
    out["studies"]["DEV-045B"] = {}
    for memory in seeds[0]["memories"]:
        reference = [r["memories"][memory]["original_tuned_threshold"] for r in seeds]
        out["studies"]["DEV-045B"][memory] = {name:contrast([r["memories"][memory][name] for r in seeds],reference)
            for name in seeds[0]["memories"][memory]}
    seeds = sorted(data["046"]["per_seed"],key=lambda r:r["seed"])
    out["studies"]["DEV-046B"] = {name:contrast([r["arms"][name] for r in seeds],
        [r["arms"]["forecast_"+name.split("_")[1]] for r in seeds]) for name in seeds[0]["arms"]}
    seeds = sorted(data["047"]["per_seed"],key=lambda r:r["seed"])
    out["studies"]["DEV-047A"] = {name:contrast([r["variants"][name]["model"] for r in seeds],
        [r["variants"][name]["public_reference"] for r in seeds]) for name in seeds[0]["variants"]}
    seeds = sorted(data["048"]["per_seed"],key=lambda r:r["seed"])
    out["studies"]["DEV-048A"] = {}
    for name in seeds[0]["evaluation"]:
        rows = [r["evaluation"][name] for r in seeds]
        if all(r["complete"] for r in rows):
            out["studies"]["DEV-048A"][name] = contrast(rows,[r["evaluation"]["exact_history"] for r in seeds])
        else:
            out["studies"]["DEV-048A"][name] = dict(complete=False,
                mean_world_coverage=statistics.mean(r["coverage"] for r in rows),total_cost_unknown=True)
    out["studies"]["DEV-048A"]["development_selected"] = [r["selected"] for r in seeds]
    return out


if __name__=="__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--evidence-root",required=True)
    ap.add_argument("--output",required=True)
    a = ap.parse_args()
    Path(a.output).write_text(json.dumps(summarize(a.evidence_root),indent=2,allow_nan=False)+"\n")
