#!/usr/bin/env python3
"""Independently reload weights and replay every stored paired episode.

Summary-only recomputation is not accepted. Checkpoint and source hashes,
complete scheduled row identity, operation traces, targets, predictions and
all paired controls must agree with an actual replay.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))

import torch  # noqa: E402
from tac_osm.development.memory_experiment import (  # noqa: E402
    EXPERIMENT_ID, TRAIN_SEEDS, EVAL_SEEDS, TRAIN_STEPS, EVAL_PAIRS,
    evaluate, summarize,
)
from tac_osm.development.noisy_memory import MemoryConfig, NoisyMemoryModel, parameter_hash  # noqa: E402


def validate(directory: Path) -> dict:
    report = json.loads((directory/'report.json').read_text())
    if report['experiment_id'] != EXPERIMENT_ID or report['status'] != 'DEVELOPMENT_ONLY' or report['confirmatory'] is not False:
        raise ValueError('incorrect experiment identity or promotion')
    expected_steps, expected_pairs = (2,1) if report['smoke'] else (TRAIN_STEPS,EVAL_PAIRS)
    if report['train_steps'] != expected_steps or report['eval_pairs'] != expected_pairs:
        raise ValueError('changed development schedule')
    from run_integrated_memory_dev import SOURCES
    if set(report['provenance']['sources']) != set(SOURCES):
        raise ValueError('incomplete source provenance')
    for name, digest in report['provenance']['sources'].items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest() != digest:
            raise ValueError(f'source hash mismatch: {name}')
    seeds = list(zip(TRAIN_SEEDS,EVAL_SEEDS))[:1 if report['smoke'] else 3]
    if len(report['checkpoints']) != len(seeds) or len(report['training']) != len(seeds):
        raise ValueError('missing checkpoint/training records')
    if len(report['per_seed']) != len(seeds):
        raise ValueError('missing per-seed results')
    replayed, train_fp = [], set()
    for (train_seed,eval_seed), checkpoint, training, seed_summary in zip(
            seeds, report['checkpoints'], report['training'], report['per_seed']):
        if (checkpoint['train_seed'],checkpoint['eval_seed']) != (train_seed,eval_seed):
            raise ValueError('incorrect seed pairing')
        if checkpoint['name'] != f'model-{train_seed}.pt':
            raise ValueError('invalid checkpoint path')
        path = directory/checkpoint['name']
        if hashlib.sha256(path.read_bytes()).hexdigest() != checkpoint['sha256']:
            raise ValueError('checkpoint byte hash mismatch')
        payload = torch.load(path, map_location='cpu', weights_only=True)
        model = NoisyMemoryModel(MemoryConfig(**payload['config'])).eval()
        model.load_state_dict(payload['state_dict'], strict=True)
        if training['seed'] != train_seed or training['steps'] != expected_steps or parameter_hash(model) != training['final_hash']:
            raise ValueError('trained checkpoint identity mismatch')
        if training['initial_hash'] == training['final_hash'] or not all(training['nonzero_finite_task_gradients'].values()) or not training['future_decision_write_gradient']:
            raise ValueError('missing training diagnostics')
        train_fp.update(training['train_fingerprints'])
        rows = evaluate(model,eval_seed,pairs=expected_pairs)
        for row in rows:
            row['train_seed'],row['eval_seed'] = train_seed,eval_seed
        # JSON normalizes nested tuple/list traces; exact numeric replay uses
        # the same documented CPU environment and thread setting.
        normalized = json.loads(json.dumps(rows))
        stored = [r for r in report['rows'] if r['train_seed'] == train_seed]
        if normalized != stored:
            raise ValueError('raw predictions/data/work traces differ from checkpoint replay')
        expected_summary = {'train_seed':train_seed,'eval_seed':eval_seed,'summary':summarize(rows)}
        if seed_summary != expected_summary:
            raise ValueError('per-seed summary mismatch')
        replayed.extend(rows)
    if len(replayed) != len(report['rows']):
        raise ValueError('extra/omitted evaluation rows')
    eval_fp = [x for row in replayed for x in row['fingerprints']]
    if len(eval_fp) != len(set(eval_fp)) or train_fp & set(eval_fp):
        raise ValueError('duplicate/overlapping evaluation episodes')
    summary = summarize(replayed)
    if summary != report['summary']:
        raise ValueError('summary mismatch')
    if summary['accuracy']['reset'] != .5 or summary['followup_accuracy']['reset'] != .5 or summary['followup_accuracy']['no_commit'] != .5:
        raise ValueError('paired history-independent ceiling failed')
    return {'status':'REPLAY_VALIDATED_DEVELOPMENT_ONLY','rows':len(replayed),'summary':summary}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('directory',type=Path)
    args = parser.parse_args()
    torch.set_num_threads(2)
    print(json.dumps(validate(args.directory),indent=2))
