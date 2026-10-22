"""ME052 original publisher and head-to-head parity checks before fitting."""
import hashlib
import json
from pathlib import Path
from scripts.validate_research_parity import check_manifest

PLAN_PATH=Path("research/mathematical-engineering/me052/FAIR_COMPARISON.json")

def canonical_hash(value,sort_keys=False):
    return hashlib.sha256(json.dumps(value,sort_keys=sort_keys,
                          separators=(",",":")).encode()).hexdigest()

def verify_before_fit(source_episodes,split_memberships,planned_schedules,model_parameter_counts):
    plan=json.loads(PLAN_PATH.read_text())
    check_manifest(plan,"preflight")
    original=[{"original_episode_index":idx,
               "source_parquet_SHA256":item["sha256"],
               "authentic_frame_count":item["length"]}
              for idx,item in sorted(source_episodes.items())]
    assert canonical_hash(original,sort_keys=True)==plan["dataset_fit_receipt_sha256"], "FIT archive checksum disagreement"
    for part in ("FIT","DEV","TEST"):
        ids=[i for i,_ in split_memberships[part]]
        assert canonical_hash(ids)==plan["frozen_episode_split_sha256"][part], "original episode cohort changed"
    assert model_parameter_counts==plan["parameter_counts"], "trainable parameter counts changed"
    assert sorted(planned_schedules)==plan["seeds"], "original paired seeds changed"
    return plan

def verify_after_fit(plan,original_records,models_frozen):
    final=dict(plan)
    final["completed_model_records"]=original_records
    final["all_frozen_original_models_before_heldout_opening"]=models_frozen
    outcome=check_manifest(final,"final")
    if not outcome["status"].startswith("PASS_OWNER_MANDATORY"):
        raise RuntimeError("actual source and training-budget parity missing")
    return final,outcome
