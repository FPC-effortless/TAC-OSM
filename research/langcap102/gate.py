"""LANGCAP102 source-only scientific readiness lint. Performs NO model fitting.

This is NOT the signed independent real-data parity gate and never authorizes
optimizer steps or protected evaluation. It checks only proposed study design.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
POLICY=HERE.parent/"REAL-WORLD-EVIDENCE-RULES.md"
PROTOCOL=HERE/"PROTOCOL.json"

class DesignGateError(ValueError):
    pass

def require(ok, message):
    if not ok: raise DesignGateError(message)

def inspect_design(p, policy_text):
    require(p.get("study_id")=="LANGCAP102", "Wrong study ID")
    require(p.get("schema")=="langcap102.capability-first.design.v1", "Wrong protocol schema")
    require(p.get("status")=="DESIGN_ONLY_NO_NEW_MODEL_FITS", "Design is not training")
    for field in ("training_enabled","dev_scoring_enabled","protected_test_access",
                  "checkpoint_promotion","independent_replication"):
        require(p.get(field) is False, "Must remain disabled: "+field)
    scale=p.get("parameter_policy",{})
    require(scale.get("policy")=="declared-target-v2", "Scaling policy not adopted")
    require(scale.get("status")=="SCALE_POLICY_APPROVED_NOT_MODEL_PREREGISTERED",
            "Scale is approved; actual model target is not registered")
    require(scale.get("size_freedom")=="NO_GLOBAL_SIZE_CEILING", "Fixed ceiling remains")
    require(scale.get("per_model_target_band_fraction")==0.05, "5% target band required")
    require(scale.get("between_arms_max_relative_gap")==0.01,
            "Cross-arm 1% parity requirement cannot be relaxed")
    require(scale.get("pretrained_deployed_weights_count") is True,
            "Pretrained weights cannot be hidden")
    require(scale.get("preregistered_trainable_target") is None and
            scale.get("preregistered_deployed_target") is None,
            "Design study cannot pretend a model has been preregistered")
    require("Model-scale policy v2" in policy_text and
            'declared-target-v2' in policy_text,
            "Research-wide governing policy not updated")
    require(len(p.get("hypotheses",[]))>=3, "Falsifiable hypotheses missing")
    require(len(p.get("capabilities",[]))>=5, "Capability families missing")
    controls=set(p.get("primary_controls",[]))
    require({"faithful_decoder_only_transformer",
             "fully_specified_recurrent_or_ssm",
             "native_tac_osm"}<=controls, "Strong primary controls missing")
    diag=set(p.get("diagnostic_controls",[]))
    require({"window4","history_reset","history_shuffle","query_only",
             "relevant_evidence_ablation"}<=diag, "Shortcut controls missing")
    datasets=p.get("datasets",[])
    require(len({d.get("id") for d in datasets})==len(datasets)>=4,
            "Duplicate or insufficient source candidates")
    primary=[d for d in datasets if d.get("role")=="POTENTIAL_PRIMARY_EXPOSED_DEV"]
    require(len(primary)>=2, "Multiple genuine source candidates required")
    for d in datasets:
        require(str(d.get("url","")).startswith("https://"), "Publisher reference required")
        if d.get("role")=="POTENTIAL_PRIMARY_EXPOSED_DEV":
            require(d.get("status")=="NOT_ACCEPTED" and
                    d.get("publisher_raw_hash")=="PENDING" and
                    d.get("licence_review")=="PENDING",
                    "Do not claim unaudited real source is qualified")
        else:
            require(d.get("role")=="DIAGNOSTIC_ONLY" and
                    str(d.get("source_type","")).startswith("synthetic"),
                    "Synthetic benchmarks are diagnostic only")
    qual=p.get("qualification",{})
    require(qual.get("max_level")=="DESIGN_ONLY", "No capability claim allowed")
    for k in ("synthetic_capability_claim","real_primary_data_received",
              "holdout_judge_access","model_superiority","commercial_readiness"):
        require(qual.get(k) is False, "No qualified evidence: "+k)
    prior=p.get("prior_evidence",{})
    require(prior.get("original_preregistered_passes")==135 and
            prior.get("original_failures")==3 and
            prior.get("recovery_is_posthoc") is True,
            "LANGMATH101 original record must remain unchanged")
    return {"status":"PASS_NO_FIT_DESIGN_AND_APPROVED_SCALE_V2",
            "training_authorized":False,"model_target_registered":False,
            "real_data_source_qualified":False,"dev_scoring_authorized":False,
            "independent_qualification":False}

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("command",choices=("design",))
    args=parser.parse_args()
    try:
        p=json.loads(PROTOCOL.read_text())
        print(json.dumps(inspect_design(p,POLICY.read_text()),sort_keys=True))
        return 0
    except (DesignGateError,ValueError,KeyError,OSError) as exc:
        print(json.dumps({"status":"FAIL_CLOSED","reason":str(exc)},sort_keys=True))
        return 2

if __name__=="__main__":
    raise SystemExit(main())
