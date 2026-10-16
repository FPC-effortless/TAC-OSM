"""Independent stdlib audit of bound frozen-checkpoint cross-evaluation."""
from __future__ import annotations
import argparse
import json
from pathlib import Path

from audit_dev024_writer_attribution import ROOT, digest, validate_rows, metrics, interval, mean, spec_and_sources

PROTOCOL = "development_protocols/DEV-025-FROZEN-CROSS-EVAL.json"
MANIFEST = "research/autoresearch/dev025-inputs.json"
SOURCES = (PROTOCOL, MANIFEST, "docs/DEV-025-FROZEN-CROSS-EVAL-PROTOCOL.md",
           "scripts/run_dev025_cross_eval.py", "scripts/audit_dev025_cross_eval.py")


def config():
    return json.loads((ROOT / PROTOCOL).read_text())


def inputs(parent, key):
    frozen = json.loads((ROOT / MANIFEST).read_text())[key]
    report, checkpoint = parent / "report.json", parent / "model.pt"
    assert digest(report) == frozen["report_sha256"]
    assert digest(checkpoint) == frozen["checkpoint_sha256"]
    r = json.loads(report.read_text())
    cohort, regime, seed = key.split("-")
    assert (r["study"], r["regime"], r["seed"], r["arm"], r["smoke"]) == (
        "DEV-023" if cohort == "DEV023" else "DEV-024", regime, int(seed),
        "masked_local" if cohort == "DEV023" else "joint", False)
    assert r["training"]["final_hash"] == frozen["parameter_hash"]
    assert r["model"]["sha256"] == frozen["checkpoint_sha256"]
    assert all(digest(ROOT / p) == h for p, h in r["source_sha256"].items())
    return r, frozen


def source_hashes(parent):
    result = dict(parent["source_sha256"])
    for p in SOURCES:
        result[p] = digest(ROOT / p)
    # The DEV-024 validator is also an actual dependency for DEV-023 inputs.
    for p in ("scripts/audit_dev024_writer_attribution.py", "contracts/DEV-024.json"):
        result[p] = digest(ROOT / p)
    return result


def summarize(records):
    c = config()
    by = {(r["cohort"], r["regime"], r["seed"], r["grid"]): r for r in records}
    expected = {(a,g,s,e) for a in c["cohorts"] for g in c["regimes"] for s in c["seeds"] for e in c["cohorts"]}
    assert len(by) == len(records) and set(by) == expected
    out = {"study": "DEV-025", "cells": len(records), "training_steps": 0,
           "adaptive_known_diagonals": True, "official_TAC_overall_success": False, "by_regime": {}}
    for regime in c["regimes"]:
        o = {"cohort_by_grid": {}, "contrasts": {}}
        for cohort in c["cohorts"]:
            for grid in c["cohorts"]:
                ms = [by[cohort,regime,s,grid]["summary"] for s in c["seeds"]]
                prediction = all(m["primary"] >= .8 and m["primary"]-m["swap_primary"] >= .1 for m in ms)
                physical = all(m["selected_failures"] <= .05 and m["selected_unnecessary_services"] <= .1 and m["selected_total_cost"] <= .8 for m in ms)
                physical &= mean([m["selected_total_cost"] for m in ms]) <= mean([m["none_total_cost"] for m in ms])
                o["cohort_by_grid"][cohort+"_on_"+grid] = {"per_seed":ms,"means":{k:mean([m[k] for m in ms]) for k in ms[0]}, "prediction_and_swap_gate":prediction,"selected_physical_gate":physical,"combined_selected_gate":prediction and physical}
        for grid in c["cohorts"]:
            o["contrasts"]["DEV024_minus_DEV023_on_"+grid] = interval([by['DEV024',regime,s,grid]['summary']['primary']-by['DEV023',regime,s,grid]['summary']['primary'] for s in c['seeds']])
        for cohort in c["cohorts"]:
            o["contrasts"]["DEV024_grid_minus_DEV023_grid_for_"+cohort] = interval([by[cohort,regime,s,'DEV024']['summary']['primary']-by[cohort,regime,s,'DEV023']['summary']['primary'] for s in c['seeds']])
        for effect in o["contrasts"].values():
            lo, hi = effect["exploratory_95_t_df4"]
            effect["meaningful_separation"] = abs(effect["mean"]) >= .05 and (lo > 0 or hi < 0)
        out["by_regime"][regime] = o
    return out


def audit(folder):
    r = json.loads((folder / "report.json").read_text())
    assert r["study"] == "DEV-025" and r["training_steps"] == 0
    key = f"{r['cohort']}-{r['regime']}-{r['seed']}"
    original, frozen = inputs(folder.parent / "original", key)
    assert r["frozen_input"] == frozen and r["source_sha256"] == source_hashes(original)
    assert r["grid"] in config()["cohorts"]
    assert r["evaluation_offset"] == config()["evaluation_offsets"][r["grid"]]
    assert json.loads((folder / "replay.json").read_text()) == {
        "verified":True,"report_sha256":digest(folder / "report.json"),"checkpoint_sha256":frozen['checkpoint_sha256'],"source_sha256":r['source_sha256']}
    spec = spec_and_sources()
    spec.update(h_levels=[512], eval_steps=4)
    validate_rows(r["rows"], r["seed"]+r["evaluation_offset"]-24000, "joint", spec)
    assert metrics(r["rows"]) == r["summary"] and r["summary"]["primary_n"] == 96
    assert r["summary"]["reset_primary"] == .5
    if r["grid"] == r["cohort"]:
        assert r["rows"] == [row for row in original["rows"] if row["condition"]["history"] == 512]
    return r


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("folders", nargs="+", type=Path)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    rs = [audit(folder) for folder in a.folders]
    summary = summarize(rs)
    a.output.write_text(json.dumps(summary, indent=2)+"\n")
    print("DEV025_INDEPENDENT_AUDIT_PASSED", len(rs), flush=True)
