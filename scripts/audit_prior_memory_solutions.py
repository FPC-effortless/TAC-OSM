"""PAR-MEM-009 source-pinned historical remedy regression audit, NO FIT.

This audit fetches each historically IMPLEMENTED old model module from its
actual pre-existing Git branch, checks the exact original Git blob SHA,
parses the original Python and verifies the protected implemented source
mechanisms. It compares those with the newly narrowed PAR-TIME neural
stream state; never substitutes any prior model for a trained new model.

Read-only Git inspection on github.com CI requires origin branch fetch.
It does not load source outcome data, run a model, retune, or overwrite
original experimental checkpoints. Structural source checks are not a
proof of mathematical equivalence or original clinical/business ability.
"""
from __future__ import annotations
import argparse
import ast
import hashlib
import json
import subprocess
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
PATH=ROOT/"research/parallel-six/PAR-MEM-009-PRIOR-SOURCE-MANIFEST.json"
OUT=ROOT/"research/parallel-six/PAR-MEM-009-SOURCE-AUDIT.json"

def git(*args, text=True, input=None):
    return subprocess.run(["git",*args], cwd=ROOT, check=True,
        stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=text,input=input).stdout

def body_at(ref,path):
    git("fetch","--no-tags","--depth=1","origin",ref)
    data=git("show",f"FETCH_HEAD:{path}")
    actual=git("hash-object","--stdin",input=data).strip()
    return data,actual

def require(src,things,role):
    ast.parse(src)
    for needle in things:
        if needle not in src:
            raise RuntimeError(f"source-pinned previous implementation {role} no longer has {needle!r}")

def review():
    cfg=json.loads(PATH.read_text())
    if cfg["study_id"]!="PAR-MEM-009-PRIOR-IMPLEMENTATION-RECONCILIATION":
        raise RuntimeError("unexpected historical research manifest")
    main,sha=body_at(cfg["canonical_master"]["branch"],cfg["canonical_master"]["path"])
    if sha!=cfg["canonical_master"]["blob_sha"]:
        raise RuntimeError("canonical real master-plan Git blob changed during audit")
    if len(main.splitlines())!=cfg["canonical_master"]["lines"]:
        raise RuntimeError("master plan archival line count drift")
    for key in ["DIAGKEY004","ME100","ME097","Memory preservation","DIAGKEY002",
                "real-world","read-only","research"]:
        if key not in main: raise RuntimeError(f"master plan provenance missing {key}")
    checks={
      "DIAGKEY004":["class AlignedAssociativeTAC","self.address(keys)",
           "self.content(values)","kind[:, j:j+1] != 2",
           "radius=4.0","self.read_gain","gated_write("],
      "ME100":["class EntitySensorWorld","owner: torch.Tensor",
           "last_hour: torch.Tensor","state.owner","state.last_hour",
           "source_admission","source_sum","source_mass",
           "diagonal_affine_prefix","associative_affine_schedule"],
      "LATFUSE004":["class FactorialFusion","partitioned_update",
           "head_weights(","common_key_loss","common_head_balance",
           '"2h-soft"',"self.soft_routing"],
      "LATFUSE007":["class SourceTimedEventQuery",
           "historic_clock_query","historic_event_logits","historic_known_clock",
           "EXPECTED_TRAINABLE=99390"],
      "LATFUSE008":["class SourceTimedNonassociativeQuery",
           "historic_event_logits","state.fast","state.slow",
           "pseudo=","read_router"],
      "LATFUSE018":["LATFUSE018","original source","H64",
           "zero-read","FAIL"]
    }
    inspected=[]
    for predecessor in cfg["predecessors"]:
        code,actual=body_at(predecessor["branch"],predecessor["path"])
        if actual!=predecessor["blob_sha"]:
            raise RuntimeError("old implementation Git blob provenance changed: "+
                               predecessor["id"])
        require(code,checks[predecessor["id"]],predecessor["id"])
        inspected.append(dict(
            original_study=predecessor["id"],ref=predecessor["branch"],
            source_path=predecessor["path"],actual_original_git_blob=actual,
            code_signature_guard="PASS",learned_real_memory_qualified=False))
    candidate=cfg["current_candidate"]
    code,sha=body_at(candidate["branch"],candidate["path"])
    if sha!=candidate["blob_sha"]:
        raise RuntimeError("original PAR-MEM008 candidate identity drift")
    module=ast.parse(code)
    typed=[n for n in module.body if isinstance(n,ast.ClassDef)
           and n.name=="TimeAwareStreamState"]
    if len(typed)!=1: raise RuntimeError("current time-stream typed state not found")
    fields=[n.target.id for n in typed[0].body
            if isinstance(n,ast.AnnAssign) and isinstance(n.target,ast.Name)]
    if fields!=["fast_sum","memory","elapsed_clock","last_write_clock",
                "write_seen","count"]:
        raise RuntimeError("simplified current time-aware persistent state contract drift")
    require(code,["class TimeAwareParallelCore","self.address(e)",
                  "self.content(e)","class TimeAwareStreamState"],"PAR-MEM008")
    report={
      "study_id":cfg["study_id"],"phase":"READ_ONLY_HISTORICAL_GIT_SOURCE_PROVENANCE",
      "canonical_master_verified_git_blob":sha if False else cfg["canonical_master"]["blob_sha"],
      "original_predecessor_source_count":len(inspected),
      "original_source_signatures_all_found":True,
      "original_source_artifacts":inspected,
      "PAR_MEM008_original_neural_source_blob":sha,
      "PAR_MEM008_state_fields":fields,
      "PAR_MEM008_no_prior_source_owner_token_in_state":"owner" not in fields,
      "PAR_MEM008_no_prior_LATFUSE_source_slots_in_state":"entities" not in fields,
      "PAR_MEM008_no_independent_fast_slow_in_state":not any(x in fields for x in ("fast","slow")),
      "original_checkpoint_source_or_weight_mutations":False,
      "synthetic_or_real_performance_scoring":False,
      "no_neural_training":True,
      "historic_solutions_reimplemented_or_integrated_in_current_candidate":False,
      "scientific_conclusion":"Prior remedy mechanisms exist and must be reused/controlled before designing or promoting a replacement; original source checks alone cannot certify real utility"
    }
    return report

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--mode",choices=("audit","replay"),required=True)
    args=p.parse_args()
    result=review()
    if args.mode=="audit":
        if OUT.exists(): raise RuntimeError("original history audit already exists")
        OUT.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    else:
        if not OUT.exists():raise RuntimeError("committed history audit missing")
        assert result==json.loads(OUT.read_text()),"original source history replay drift"
    print("PAR_MEM_009_HISTORICAL_SOURCE_REUSE_AUDIT="+json.dumps({
        "status":"PASS",
        "historically_implemented_source_modules":len(result["original_source_artifacts"]),
        "canonical_master_source_SHA_pinned":True,
        "current_narrow_state_shape_confirmed":True,
        "no_training_or_model_promotion":True},sort_keys=True),flush=True)

if __name__=="__main__":
    main()
