"""Execute and audit the registered development program from supplied DEV-044A artifacts.

Writes outside the checkout, never overwrites an evidence directory, and never
merges, deploys, tunes on evaluation outcomes or launches paid cloud resources.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def call(script,*args):
    subprocess.run([sys.executable,str(ROOT/"scripts"/script),*map(str,args)],cwd=ROOT,check=True)


def run(parent_root,output):
    parent_root,output = Path(parent_root).resolve(),Path(output).resolve()
    if output.is_relative_to(ROOT):
        raise ValueError("place evidence outside the source checkout to preserve source-lock checks")
    output.mkdir(parents=True,exist_ok=False)
    call("audit_reservoir_scaling_dev044.py",*[parent_root/str(s) for s in (4401,4402,4403)],"--output",output/"dev044-audit")
    studies = {"045":(4501,4502,4503),"046":(4601,4602,4603),"047":(4701,4702,4703),"048":(4801,4802,4803)}
    runners = {"045":"run_frozen_policy_dev045.py","046":"run_joint_training_dev046.py",
               "047":"run_local_generalization_dev047.py","048":"run_verified_routing_dev048.py"}
    audits = {"045":"audit_frozen_policy_dev045.py","046":"audit_joint_training_dev046.py",
              "047":"audit_local_generalization_dev047.py","048":"audit_verified_routing_dev048.py"}
    for study,seeds in studies.items():
        parent_args = []
        if study=="045":
            parent_args = ["--parent-root",parent_root,"--parent-audit",output/"dev044-audit/analysis.json"]
        elif study=="048":
            parent_args = ["--parent-root",output/"dev046","--parent-audit",output/"dev046/analysis.json"]
        for seed in seeds:
            call(runners[study],"--seed",seed,"--output",output/("dev"+study)/str(seed),*parent_args)
        audit_args = ["--parent-root",parent_root] if study=="045" else []
        call(audits[study],*[output/("dev"+study)/str(s) for s in seeds],
             "--output",output/("dev"+study)/"analysis.json",*audit_args)
    (output/"completion.json").write_text(json.dumps(dict(status="DEVELOPMENT_PROGRAM_REPLAYED",
        studies=list(studies),confirmatory_evidence=False,full_architecture_superiority_established=False),indent=2)+"\n")


if __name__=="__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--parent-root",required=True,help="DEV-044A artifacts normalized as root/4401,4402,4403")
    ap.add_argument("--output",required=True)
    a = ap.parse_args()
    run(a.parent_root,a.output)
