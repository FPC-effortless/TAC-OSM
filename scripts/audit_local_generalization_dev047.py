"""DEV-047A checkpoint, public reference, permutation and physical replay audit."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/"src"),str(ROOT/"scripts")]
import torch
from tac_osm.development.local_memory_variants import build_variant
from run_local_generalization_dev047 import cohort, predict, physical_rows, totals, training_cases, VARIANTS, PROTOCOL
from run_frozen_policy_dev045 import digest
from run_reservoir_scaling_dev044 import THRESHOLDS


def audit(folder):
    folder = Path(folder)
    r = json.loads((folder/"report.json").read_text())
    assert not r["smoke"] and r["study"]=="DEV-047A"
    assert r["protocol_sha256"]==digest(PROTOCOL) and set(r["variants"])==set(VARIANTS)
    seed = r["seed"]
    training = [c.fingerprint for step in range(72) for c in training_cases(seed,step)]
    calibration = cohort(seed,"calibration")
    evaluation = cohort(seed,"evaluation")
    cal_fps = {c.fingerprint for pair,_ in calibration for c in pair}
    eval_fps = {c.fingerprint for pair,_ in evaluation for c in pair}
    assert len(training)==len(set(training))==144
    assert not cal_fps&eval_fps and not set(training)&cal_fps and not set(training)&eval_fps
    summary = {}
    for variant,data in r["variants"].items():
        ck = folder/(variant+".pt")
        assert digest(ck)==data["checkpoint_sha256"]
        state = torch.load(ck,weights_only=True,map_location="cpu")
        model = build_variant(variant,seed)
        model.load_state_dict(state)
        model.eval()
        assert data["train_fingerprints"]==training and data["curve"][-1]["step"]==72
        cal = []
        for (cases,condition),saved in zip(calibration,data["calibration"]):
            p = predict(model,cases)
            assert saved==dict(condition=condition,rows=physical_rows(cases,p,data["threshold"]))
            cal.append((cases,p))
        expected = min((sum(totals(physical_rows(cs,p,t))["total_cost"] for cs,p in cal),abs(t-.25),t)
                       for t in THRESHOLDS)[-1]
        assert expected==data["threshold"]
        assert len(data["evaluation"])==len(evaluation)==144
        all_rows = []
        for (cases,condition),saved in zip(evaluation,data["evaluation"]):
            scaled = build_variant(variant,seed,entities=condition["entities"])
            scaled.load_state_dict(state)
            scaled.eval()
            p = predict(scaled,cases)
            rows = physical_rows(cases,p,data["threshold"])
            assert saved["condition"]==condition and saved["rows"]==rows
            assert saved["summary"]==totals(rows) and saved["reference"]==totals(rows,"public_reference")
            assert saved["last_value"]==totals(rows,"last_value")
            assert saved["state_scalars"]==condition["entities"]*25
            all_rows.extend(rows)
        summary[variant] = dict(model=totals(all_rows),public_reference=totals(all_rows,"public_reference"),
                                last_value=totals(all_rows,"last_value"))
    return dict(seed=seed,source=r["source"],report_sha256=digest(folder/"report.json"),variants=summary)


if __name__=="__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("folders",nargs=3)
    ap.add_argument("--output",required=True)
    a = ap.parse_args()
    torch.set_num_threads(1)
    results = [audit(f) for f in a.folders]
    assert {r["seed"] for r in results}=={4701,4702,4703}
    assert len({r["source"] for r in results})==1
    Path(a.output).write_text(json.dumps(dict(study="DEV-047A",status="CHECKPOINT_AND_PHYSICAL_REPLAY_PASSED_DEVELOPMENT_ONLY",
        per_seed=results,confirmatory_claim=False),indent=2)+"\n")
    print("DEV-047A checkpoint and physical replay passed for all three seeds")
