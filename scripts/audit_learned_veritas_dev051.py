"""DEV-051 independently replay native Veritas outcomes and frozen policy decisions.

This audit reconstructs public cohorts, checks source and checkpoint identities,
repeats calibration/gating and re-executes every action. It does NOT redo SGD.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import statistics
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/"src"),str(ROOT/"scripts")]
import torch
from tac_osm.development.learned_veritas_policy import (
    PublicActionPolicy,public_cohort,digest,aggregate)
from run_learned_veritas_dev051 import PROTOCOL,evaluate,select_gate


def audit_seed(directory):
    directory=Path(directory)
    report=json.loads((directory/"report.json").read_text())
    proto=json.loads(PROTOCOL.read_text())
    seed=report["seed"]
    assert seed in proto["data"]["model_seeds"] and not report["smoke"]
    assert report["status"]=="PUBLIC_LEARNED_POLICY_DEVELOPMENT_NOT_CONFIRMATORY"
    assert report["protocol_sha256"]==digest(PROTOCOL)
    assert report["veritas_source"]==proto["environment"]["commit"]
    assert report["model_fit"]["optimizer_updates_per_arm"]==160
    assert report["model_fit"]["teacher_examples"]>0
    assert report["model_fit"]["student_encountered_examples"]>0
    splits={}
    for key,kind in (("teacher_worlds","teacher"),("onpolicy_worlds","onpolicy"),
                     ("calibration_worlds","calibration"),("evaluation_worlds","evaluation")):
        spec=proto["data"][key]
        cases=list(public_cohort(spec["seeds"],spec["worlds_per_seed"]))
        ids=[c.public_payload()["episode_id"] for _,_,c in cases]
        assert ids==report["split_ids"][kind] and len(ids)==len(set(ids))
        splits[kind]=cases
    sets=[set(report["split_ids"][k]) for k in ("teacher","onpolicy","calibration","evaluation")]
    assert all(not sets[i]&sets[j] for i in range(len(sets)) for j in range(i))
    assert len(splits["evaluation"])==128
    networks={}
    for name in ("bc","dagger"):
        saved=directory/(name+".pt")
        assert digest(saved)==report["checkpoint_sha256"][name]
        model=PublicActionPolicy(seed)
        model.load_state_dict(torch.load(saved,map_location="cpu",weights_only=True),strict=True)
        model.eval()
        networks[name]=model
    threshold,candidates=select_gate(networks["dagger"],splits["calibration"])
    assert threshold==report["confidence_threshold"]
    assert candidates==report["calibration_candidates"]
    reference=[
        ("inspect_verify_retry",None,None),("inspect_once",None,None),
        ("stale_record",None,None),("public_belief_rule",None,None),
        ("behavior_cloning",networks["bc"],None),
        ("on_policy_aggregation",networks["dagger"],None),
        ("confidence_gated_aggregation",networks["dagger"],threshold)]
    audited={}
    for name,model,gate in reference:
        rows=evaluate(splits["evaluation"],name,model,gate)
        known=report["policies"][name]["rows"]
        assert len(rows)==len(known)==128
        for new,old in zip(rows,known):
            assert (new["seed"],new["index"],new["world"])==(
                    old["seed"],old["index"],old["world"])
            for k in ("failures","complete","total_cost","blocked","tool_calls",
                      "sensing_calls","fallbacks","decisions"):
                assert new[k]==old[k],f"physical result drift: {seed} {name} {k}"
            def normalized_actions(v):
                # Logit accumulation can move confidence by the final ulp,
                # but must never change the chosen action or native receipt.
                return [{k:item[k] for k in ("action","parameters","result") if k in item}
                        for item in v]
            assert normalized_actions(new["actions"])==normalized_actions(old["actions"])
            assert new["verification"]==old["verification"]
        summary=aggregate(rows)
        for k,v in summary.items():
            assert abs(v-report["policies"][name]["summary"][k])<1e-10
        audited[name]=summary
    return dict(seed=seed,source=report["tac_source"],policies=audited,
                selected_gate=threshold,checkpoint_hashes=report["checkpoint_sha256"],
                train_samples=report["model_fit"]["teacher_examples"],
                student_encountered_samples=report["model_fit"]["student_encountered_examples"])


def run(paths,output):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    results=[audit_seed(p) for p in paths]
    assert sorted(x["seed"] for x in results)==[7101,7102,7103]
    assert len({r["source"] for r in results})==1
    names=list(results[0]["policies"])
    aggregate_result={}
    for name in names:
        scores=[r["policies"][name] for r in results]
        costs=[z["cost_per_entity"] for z in scores]
        aggregate_result[name]=dict(mean_cost_per_entity=statistics.mean(costs),
            seed_sd=statistics.stdev(costs),
            residual_faults=sum(z["residual_faults"] for z in scores),
            complete_worlds=sum(z["complete_worlds"] for z in scores),
            blocked_actions=sum(z["blocked_actions"] for z in scores),
            average_fallback_fraction=statistics.mean(z["fallback_fraction"] for z in scores))
    reference=aggregate_result["inspect_verify_retry"]
    robust={}
    for name in ("behavior_cloning","on_policy_aggregation","confidence_gated_aggregation"):
        x=aggregate_result[name]
        robust[name]=(x["residual_faults"]<=reference["residual_faults"] and
                      x["blocked_actions"]<=reference["blocked_actions"] and
                      x["mean_cost_per_entity"]<reference["mean_cost_per_entity"])
    report=dict(study="DEV-051A",status="AUDITED_ADAPTIVE_DEVELOPMENT_NOT_CONFIRMATORY",
                per_model_seed=results,summary=aggregate_result,
                passed_strict_reliability_and_cost_guard=robust,
                audit_checks="independent checkpoint inference, calibration replay, native Veritas actions and verifier",
                independent_optimizer_replay=False,
                learned_TAC_OSM_world_state=False,
                external_domain_shift_confirmation=False)
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    (output/"analysis.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    print(json.dumps(report,indent=2,allow_nan=False),flush=True)


if __name__=="__main__":
    a=argparse.ArgumentParser()
    a.add_argument("seed_directories",nargs=3)
    a.add_argument("--output",required=True)
    x=a.parse_args()
    run(x.seed_directories,x.output)
