"""DEV-052A pinned DEV-051 frozen-policy transfer; no new model training."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
import torch
from tac_osm.development.learned_veritas_policy import PublicActionPolicy
from tac_osm.development.noisy_veritas_transfer import run_ood,study_cohort,summarize
from tac_osm.development.learned_veritas_policy import digest

PROTOCOL=ROOT/"research/dev052/study-protocol.json"
POLICIES=("bayesian_dp","legacy_verify_retry","all_three","wait",
          "bc_frozen","dagger_frozen","dagger_old_gate_frozen")


def verify_sources(veritas_root):
    import investigation_world
    source=json.loads(PROTOCOL.read_text())
    root=Path(veritas_root).resolve()
    assert Path(investigation_world.__file__).resolve().is_relative_to(root)
    assert subprocess.check_output(["git","rev-parse","HEAD"],cwd=root,text=True).strip()==source["source"]["veritas_commit"]
    assert not subprocess.check_output(["git","status","--porcelain"],cwd=root,text=True).strip()
    assert not subprocess.check_output(["git","status","--porcelain","--untracked-files=no"],
                                       cwd=ROOT,text=True).strip()


def load_frozen(seed,source):
    folder=Path(source)/f"DEV051-RAW-seed{seed}-development"
    report=json.loads((folder/"report.json").read_text())
    assert report["study"]=="DEV-051A" and report["seed"]==seed and not report["smoke"]
    assert report["tac_source"]=="d514fb5bd34b2de3932925cd1d1beb3b480c85f3"
    nets={}
    for name in ("bc","dagger"):
        path=folder/(name+".pt")
        assert digest(path)==report["checkpoint_sha256"][name]
        model=PublicActionPolicy(seed)
        model.load_state_dict(torch.load(path,map_location="cpu",weights_only=True),strict=True)
        model.eval()
        nets[name]=model
    # The original confidence gate was selected before DEV-051 evaluation,
    # NOT refitted using DEV-052 outcomes.
    assert report["confidence_threshold"]==0.
    return nets,dict(original_sha256=report["checkpoint_sha256"],
                     original_gate=report["confidence_threshold"],
                     original_protocol_sha256=report["protocol_sha256"])


def evaluate(worlds,nets,gate,smoke=False):
    outputs={p:[] for p in POLICIES}
    for seed,index,world in worlds:
        for policy in POLICIES:
            model=nets["bc"] if policy=="bc_frozen" else nets["dagger"] if policy.startswith("dagger") else None
            threshold=gate if policy=="dagger_old_gate_frozen" else None
            case=run_ood(world,seed,index,policy,model,threshold)
            outputs[policy].append(dict(seed=seed,index=index,**case))
    return {name:dict(summary=summarize(rows),rows=rows) for name,rows in outputs.items()}


def run(seed,original,output,veritas_root,smoke=False):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    verify_sources(veritas_root)
    proto=json.loads(PROTOCOL.read_text())
    assert seed in proto["source"]["trained_seeds"]
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    nets,checkpoint=load_frozen(seed,original)
    def cohort(kind):
        spec=proto["data"]
        keys=spec["calibration_seeds"] if kind=="calibration" else spec["evaluation_seeds"]
        count=spec["calibration_worlds_per_seed"] if kind=="calibration" else spec["evaluation_worlds_per_seed"]
        if smoke:keys=keys[:1];count=2
        return list(study_cohort(keys,count))
    cal=cohort("calibration");evaluation=cohort("evaluation")
    ids={k:[ep.episode_id for _,_,ep in pool] for k,pool in (("calibration",cal),("evaluation",evaluation))}
    assert len(set(ids["calibration"]))==len(ids["calibration"])
    assert len(set(ids["evaluation"]))==len(ids["evaluation"])
    assert set(ids["calibration"]).isdisjoint(ids["evaluation"])
    start=time.monotonic()
    cal_result=evaluate(cal,nets,checkpoint["original_gate"])
    # Frozen pre-registered policies; no DEV-052 calibration thresholds were
    # changed after measuring the development cohort.
    test_result=evaluate(evaluation,nets,checkpoint["original_gate"])
    report=dict(study="DEV-052A",status="FROZEN_TRANSFER_DEVELOPMENT_NOT_CONFIRMATORY",
        seed=seed,smoke=smoke,source=subprocess.check_output(["git","rev-parse","HEAD"],
            cwd=ROOT,text=True).strip(),protocol_sha256=digest(PROTOCOL),
        dev051_checkpoint=checkpoint,split_ids=ids,
        calibration={k:v["summary"] for k,v in cal_result.items()},
        evaluation=test_result,
        evaluation_seconds=time.monotonic()-start,
        fixed_procedural_reference="public Bayesian DP supplied the known generative probabilities",
        frozen_source_limited_to_two_repair_attempts=True)
    (output/"report.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    print(json.dumps({k:v["summary"] for k,v in test_result.items()},indent=2),flush=True)


if __name__=="__main__":
    a=argparse.ArgumentParser()
    a.add_argument("--seed",type=int,required=True)
    a.add_argument("--dev051",required=True)
    a.add_argument("--output",required=True)
    a.add_argument("--veritas-root",required=True)
    a.add_argument("--smoke",action="store_true")
    v=a.parse_args();run(v.seed,v.dev051,v.output,v.veritas_root,v.smoke)
