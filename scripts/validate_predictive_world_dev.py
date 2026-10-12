"""Fail-closed checkpoint-to-raw-row replay for predictive DEV-006.

This validates evaluation evidence and training input provenance, but does not
retrain checkpoints or constitute an independent causal benchmark.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from tac_osm.development.predictive_world import ARMS, PredictiveWorldModel, make_world_batch  # noqa: E402
from tac_osm.development.noisy_memory import parameter_hash  # noqa: E402
from tac_osm.development.protection_benchmark import seed_for  # noqa: E402
from run_predictive_world_dev import EVAL_SEEDS, TRAIN_SEEDS, measure_pair, summarize  # noqa: E402


def _equal(expected, observed, path='root'):
    """Compare structured outputs, tolerating reproducible float differences."""
    if type(expected) is not type(observed):
        raise ValueError(f'{path}: type mismatch')
    if isinstance(expected, dict):
        if expected.keys()!=observed.keys():
            raise ValueError(f'{path}: dictionary keys differ')
        for key in expected:
            _equal(expected[key], observed[key], f'{path}.{key}')
    elif isinstance(expected, list):
        if len(expected)!=len(observed):
            raise ValueError(f'{path}: list lengths differ')
        for i,(a,b) in enumerate(zip(expected,observed)):
            _equal(a,b,f'{path}[{i}]')
    elif isinstance(expected,float):
        if not math.isfinite(expected) or not math.isfinite(observed):
            raise ValueError(f'{path}: nonfinite number')
        if not math.isclose(expected,observed,rel_tol=1e-5,abs_tol=1e-6):
            raise ValueError(f'{path}: numerical mismatch {expected} != {observed}')
    elif expected != observed:
        raise ValueError(f'{path}: mismatch {expected} != {observed}')


def _training_batch_digest(batch):
    return hashlib.sha256(batch.addresses.numpy().tobytes()+
        batch.values.numpy().tobytes()+batch.query.numpy().tobytes()+
        batch.current.numpy().tobytes()+batch.action.numpy().tobytes()).hexdigest()


def validate(directory: Path) -> dict:
    torch.set_num_threads(2)
    report=json.loads((directory/'report.json').read_text())
    if report.get('experiment_id')!='PREDICTIVE-WORLD-DEV-006' or \
       report.get('status')!='DEVELOPMENT_ONLY_NOT_CONFIRMATORY':
        raise ValueError('wrong experiment/status')
    smoke=report.get('smoke')
    if type(smoke) is not bool:
        raise ValueError('smoke declaration absent')
    expected_seeds=TRAIN_SEEDS[:1] if smoke else TRAIN_SEEDS
    if len(report['records'])!=len(expected_seeds):
        raise ValueError('missing or duplicate training seed')
    for record,seed,eval_seed in zip(report['records'],expected_seeds,
                                      EVAL_SEEDS[:1] if smoke else EVAL_SEEDS):
        if record['seed']!=seed or record['eval_seed']!=eval_seed:
            raise ValueError('seed association mismatch')
        data=record['training']
        if data['training_seed']!=seed or set(data['arms'])!=set(ARMS):
            raise ValueError('wrong training record')
        steps=2 if smoke else 120
        hashes, fingerprints=[],[]
        for step in range(steps):
            batch=make_world_batch(seed,step,smoke=smoke)
            hashes.append(_training_batch_digest(batch))
            fingerprints.extend(batch.fingerprints)
        _equal(data['training_batch_hashes'],hashes)
        _equal(data['training_fingerprints'],fingerprints)
        torch.manual_seed(seed)
        initial=parameter_hash(PredictiveWorldModel())
        checkpoints=torch.load(directory/f'weights-{seed}.pt',map_location='cpu',
                               weights_only=True)
        if set(checkpoints)!=set(ARMS):
            raise ValueError('missing arm checkpoint')
        eval_fingerprints=set()
        for arm in ARMS:
            entry=data['arms'][arm]
            if entry['initial_hash']!=initial or entry['final_hash']==initial:
                raise ValueError('initial checkpoint integrity')
            model=PredictiveWorldModel()
            model.load_state_dict(checkpoints[arm],strict=True)
            model.eval()
            if parameter_hash(model)!=entry['final_hash']:
                raise ValueError('final checkpoint integrity')
            rows=record['rows'][arm]
            expected_count=4 if smoke else 3*3*2*2*2*2
            if len(rows)!=expected_count:
                raise ValueError('incomplete evaluation grid')
            for row in rows:
                c=row['condition']
                eval_row=measure_pair(model,seed_for('predworld-eval',
                    eval_seed,c['H'],c['D'],c['hard_fraction'],c['noise'],
                    c['task'],c['action']),c['H'],c['D'],c['hard_fraction'],
                    c['noise'],c['task'],c['action'])
                _equal(eval_row,row)
                eval_fingerprints.update(row['visible_fingerprints'])
            _equal(summarize(rows),record['summaries'][arm])
            if record['summaries'][arm]['reset_accuracy']!=0.5:
                raise ValueError('reset control not at paired ceiling')
        if eval_fingerprints.intersection(fingerprints):
            raise ValueError('training/evaluation leakage')
        # Since the model never sees oracle slot and task role, model-specific
        # evidence must still use identical evaluation episodes in every arm.
        base=[row['visible_fingerprints'] for row in record['rows'][ARMS[0]]]
        for arm in ARMS[1:]:
            _equal(base,[row['visible_fingerprints'] for row in record['rows'][arm]])
    return {'status':'REPLAY_VALIDATED_DEVELOPMENT_ONLY',
            'seed_count':len(report['records']),
            'rows_per_arm_per_seed':4 if smoke else 144,
            'warning':'Replays evaluation from checkpoints; does not retrain models.'}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('directory',type=Path)
    args=parser.parse_args()
    print(json.dumps(validate(args.directory),indent=2))


if __name__=='__main__':
    main()
