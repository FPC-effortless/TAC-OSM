#!/usr/bin/env python3
"""Model02 resource economics arithmetic: source-owned censored DEV alerts, NO ROI.

For k reviewed drives, each review costs c and at most P known / U unknown
future 7-day recorded failures might be prevented. At hypothetical 100%
intervention effectiveness, saved loss per prevented failure must exceed
k/P*c (optimistic unknown=negative) or k/(P+U)*c (unknown=positive).
This is an action-value *sensitivity bound*, NOT observed preventability,
actual review cost, business ROI, prospective risk skill or TAC neural result.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/"research/source-evidence/MODEL02-CENSOR-INCLUSIVE-001-ORIGINAL.json"
SOURCE_SHA="7f5a53e79b16e695b4b034955806bb3bb194534b169f6eca460dfd3dc48096fe"

class EconomicError(ValueError):
    pass

def ratio(numerator,denominator):
    return numerator/denominator if denominator>0 else None

def compute(receipt):
    if (not isinstance(receipt,dict)
        or receipt.get("schema")!="model02-original-censor-inclusive-first-decision-bounds-v1"
        or receipt.get("source_zip_sha256")!=SOURCE_SHA
        or receipt.get("first_eligible_distinct_drives")!=26311
        or receipt.get("first_decision_outcomes")!={"POSITIVE":31,"NEGATIVE":25618,"CENSORED":662}
        or receipt.get("fitting_or_calibration_during_evaluation") is not False
        or receipt.get("independent_evaluation") is not False):
        raise EconomicError("not the original source-backed first eligible DEV evaluation")
    budgets=receipt.get("budgets")
    if not isinstance(budgets,list) or len(budgets)!=2:
        raise EconomicError("source review budget set unexpectedly changed")
    outputs=[]
    for item in budgets:
        k=item.get("review_capacity_drives")
        if type(k)!=int or k<=0:
            raise EconomicError("invalid number of source physical drives reviewed")
        models=item.get("models")
        if set(models or {})!={"eight_feature_logistic","two_feature_lookup"}:
            raise EconomicError("original model comparison changed")
        output={"reviewed_drives":k,"review_fraction":item["review_fraction"],"models":{}}
        for name,v in models.items():
            pos=v["known_positive"]
            neg=v["known_negative"]
            unknown=v["unknown_censored"]
            if any(type(z)!=int or z<0 for z in (pos,neg,unknown)) or pos+neg+unknown!=k:
                raise EconomicError("invalid original source outcome partition")
            if v["reviewed_drives"]!=k:
                raise EconomicError("models not compared at the same review capacity")
            output["models"][name]={
                "reviews":k,"known_source_positive":pos,
                "known_source_negative":neg,"unknown_censored":unknown,
                "potential_preventable_failure_count_at_perfect_effect_min":pos,
                "potential_preventable_failure_count_at_perfect_effect_max":pos+unknown,
                "benefit_per_prevented_event_over_review_cost_break_even_best_case":
                    ratio(k,pos+unknown),
                "benefit_per_prevented_event_over_review_cost_break_even_if_all_unknown_negative":
                    ratio(k,pos),
                "break_even_for_actual_intervention_effectiveness":(
                    "divide either threshold by independently proven prevented-failure fraction, if positive"),
                "actual_prevented_failures":"UNKNOWN",
                "actual_cost_of_technician_review":"UNKNOWN",
                "economic_net_benefit":"NOT_MEASURABLE"
            }
        outputs.append(output)
    return {
        "schema":"model02-backblaze-exposed-action-cost-threshold-v1",
        "source_zip_sha256":SOURCE_SHA,
        "denominator":"one first eligible original physical drive; all unknown labels retained",
        "unit":"hypothetical expected avoided failure loss / cost per individual technician review",
        "original_first_eligible_drive_count":26311,
        "review_budgets":outputs,
        "assumed_action_success_for_displayed_threshold":1.0,
        "assumed_all_known_recorded_failures_preventable":True,
        "source_has_verified_prevention_rate":False,
        "actual_source_asof_available_at_verified":False,
        "actual_business_costs_measured":False,
        "fitted_any_model":False,
        "independent_evaluation":False,
        "qualified_operator_recommendation":False,
        "evidence_tier":"POST_EXPOSURE_DESCRIPTIVE_HYPOTHETICAL_BREAK_EVEN_ONLY",
        "warnings":[
            "historical publisher failure flag does not prove replacement would have prevented loss",
            "assume all known positives preventable and perfect action, so actual thresholds can be much higher",
            "unknown/censored drive outcomes lead to best/worst *label* scenario bounds, not credible uncertainty intervals",
            "salaries, intervention expenses, adverse side effects, hardware prices and customer benefits are not measured",
            "not a causal effect or evidence of saved drives, revenue, ROI, paying customers or TAC memory",
            "no protected TEST, newly trained model or independent source used"
        ]
    }

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args(argv)
    try:
        if a.out.exists():
            raise EconomicError("refuse to overwrite original decision receipt")
        evidence=json.loads(SOURCE.read_bytes())
        data=compute(evidence)
        a.out.parent.mkdir(parents=True,exist_ok=True)
        a.out.write_text(json.dumps(data,sort_keys=True,indent=2)+"\n")
        print(json.dumps(data,sort_keys=True,allow_nan=False))
        return 0
    except (OSError,EconomicError,TypeError,ValueError,KeyError) as e:
        print(json.dumps({"status":"BLOCKED_OR_FAILED","reason":str(e)},sort_keys=True),file=sys.stderr)
        return 2

if __name__=="__main__":
    sys.exit(main())
