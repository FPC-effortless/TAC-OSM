#!/usr/bin/env python3
"""Run inference using an original, frozen ME004 real-data experimental checkpoint.

Usage:
python -m scripts.me004_predict --report results/me004/results.json \
 --checkpoint results/me004/checkpoints/tac_two_timescale-240401.pt \
 --observations observations.json

observations.json is a chronological JSON list of exactly 32 objects, each with
the 12 physical sensor names from FEATURES. No measurement after prediction time
may appear in the input. Horizons are OBSERVATION STEPS, not guaranteed minutes.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
import torch
from scripts.me004_native_real_forecast import FEATURES,HIST,HORIZONS,make_model

def forecast(report_file,checkpoint_file,observations_file):
    report=json.loads(Path(report_file).read_text())
    if report['study']!='ME004' or report.get('architecture_promoted') is not False:
        raise ValueError('expected original unpromoted ME004 research report')
    cohort=report['cohort']
    mean=np.asarray(cohort['mean_train_only'],dtype=np.float32)
    scale=np.asarray(cohort['std_train_only'],dtype=np.float32)
    if not (np.isfinite(mean).all() and np.isfinite(scale).all() and (scale>0).all()):
        raise ValueError('invalid training-only normalization')
    rows=json.loads(Path(observations_file).read_text())
    if not isinstance(rows,list) or len(rows)!=HIST:
        raise ValueError(f'expect {HIST} chronological sensor records')
    try:
        physical=np.array([[float(row.get(f,float('nan'))) for f in FEATURES] for row in rows],dtype=np.float32)
    except (AttributeError,ValueError,TypeError) as exc:
        raise ValueError('sensor records must map real sensor names to numeric values') from exc
    mask=np.isfinite(physical)
    if not mask[-1].all():
        raise ValueError('most recent record must have all target sensor channels')
    scaled=np.clip(np.nan_to_num((physical-mean)/scale,nan=0.0,posinf=0,neginf=0),-10,10)
    model_x=np.concatenate([scaled,mask.astype(np.float32)],axis=1)
    ck=torch.load(checkpoint_file,map_location='cpu',weights_only=True)
    kind=ck['kind']
    declared={(f['model'],f['seed'],f['checkpoint_sha256']) for f in report['fits']}
    import hashlib
    digest=hashlib.sha256(Path(checkpoint_file).read_bytes()).hexdigest()
    if (kind,ck['seed'],digest) not in declared:
        raise ValueError('checkpoint does not match original recorded score/weight manifest')
    model=make_model(kind)
    model.load_state_dict(ck['state_dict'],strict=True)
    model.eval()
    with torch.inference_mode():
        delta=model(torch.from_numpy(model_x[None]))[0].numpy()
    forecast_raw=physical[-1][None,:]+delta*scale[None,:]
    if not np.isfinite(forecast_raw).all():
        raise ValueError('nonfinite forecast')
    return {
      'research_model':kind,'checkpoint_sha256':digest,'training_seed':ck['seed'],
      'predictions':{str(h):dict(zip(FEATURES,[float(v) for v in forecast_raw[i]]))
                     for i,h in enumerate(HORIZONS)},
      'horizons_are_observation_steps':True,
      'trained_on_authentic_operational_data':True,
      'independent_future_evaluation_passed':False,
      'not_authorized_for_repair_actions_or_commercial_deployment':True}

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--report',required=True)
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--observations',required=True)
    a=p.parse_args()
    print(json.dumps(forecast(a.report,a.checkpoint,a.observations),indent=2,sort_keys=True))
if __name__=='__main__':main()
