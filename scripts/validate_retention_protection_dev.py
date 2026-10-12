#!/usr/bin/env python3
"""Replay every protection arm and verify matched data/initialization controls."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))

import torch  # noqa: E402
from tac_osm.development.protected_memory import ProtectionModel  # noqa: E402
from tac_osm.development.noisy_memory import MemoryConfig,parameter_hash  # noqa: E402
from tac_osm.development.protection_experiment import (  # noqa: E402
    ID,protocol,matched_models,training_batch,evaluate,summarize,
)
from tac_osm.development.protection_benchmark import fingerprint  # noqa: E402
from run_retention_protection_dev import SOURCES,digest  # noqa: E402


def validate(directory: Path) -> dict:
    report = json.loads((directory/'report.json').read_text())
    p = protocol()
    if report['experiment_id']!=ID or report['status']!='DEVELOPMENT_ONLY' or report['confirmatory'] is not False:
        raise ValueError('wrong identity or promoted result')
    if report['protocol']!=p or set(report['source_sha256'])!=set(SOURCES):
        raise ValueError('protocol/source coverage mismatch')
    for name,expected in report['source_sha256'].items():
        if digest(ROOT/name)!=expected:
            raise ValueError('source hash mismatch')
    seeds = list(zip(p['model_seeds'],p['evaluation_seeds']))[:1 if report['smoke'] else 3]
    if len(report['records'])!=len(seeds) or len(report['training'])!=len(seeds) or len(report['checkpoints'])!=len(seeds)*len(p['arms']):
        raise ValueError('missing/extra model seeds or checkpoints')
    steps = 2 if report['smoke'] else p['training']['steps']
    replayed,all_eval = [],set()
    for (seed,eval_seed),training,stored in zip(seeds,report['training'],report['records']):
        if training['seed']!=seed or training['steps']!=steps or stored['model_seed']!=seed or stored['evaluation_seed']!=eval_seed:
            raise ValueError('wrong seed/schedule')
        if set(training['arms'])!=set(p['arms']) or set(stored['rows'])!=set(p['arms']):
            raise ValueError('missing arm')
        initial = matched_models(seed)
        # Rebuild every training data fingerprint independently of its stored
        # attestation. Training weights/gradient histories are not retrained.
        expected_train = [fingerprint(e) for step in range(steps) for e in training_batch(seed,step)]
        if training['train_fingerprints']!=expected_train or len(expected_train)!=len(set(expected_train)):
            raise ValueError('training data identity/duplicates mismatch')
        rows,shared_fp = {},None
        for arm in p['arms']:
            entries = [c for c in report['checkpoints'] if c['model_seed']==seed and c['arm']==arm]
            if len(entries)!=1:
                raise ValueError('checkpoint coverage mismatch')
            entry = entries[0]
            if entry['evaluation_seed']!=eval_seed or entry['name']!=f'{arm}-{seed}.pt':
                raise ValueError('wrong checkpoint path/seed')
            path = directory/entry['name']
            if digest(path)!=entry['sha256']:
                raise ValueError('checkpoint byte hash mismatch')
            payload = torch.load(path,map_location='cpu',weights_only=True)
            if payload['mode']!=arm:
                raise ValueError('mode mismatch')
            model = ProtectionModel(MemoryConfig(**payload['config']),arm).eval()
            model.load_state_dict(payload['state_dict'],strict=True)
            t = training['arms'][arm]
            if t['initial_hash']!=parameter_hash(initial[arm]) or t['final_hash']!=parameter_hash(model) or t['initial_hash']==t['final_hash']:
                raise ValueError('initial/trained checkpoint mismatch')
            if t['steps']!=steps or not all(t['gradient_seen'].values()) or t['gate_gradient_seen']!=(arm=='learned_protection'):
                raise ValueError('training diagnostics missing')
            if t['active_trainable_parameters']!=sum(x.numel() for x in model.parameters() if x.requires_grad):
                raise ValueError('active parameter count mismatch')
            measured = evaluate(model,eval_seed,smoke=report['smoke'])
            if json.loads(json.dumps(measured))!=stored['rows'][arm]:
                raise ValueError('raw predictions, controls or gate traces differ from replay')
            fp = [x for r in measured for x in r['fingerprints']]
            if len(fp)!=len(set(fp)) or set(expected_train)&set(fp):
                raise ValueError('duplicate/overlapping evaluation cases')
            if shared_fp is not None and fp!=shared_fp:
                raise ValueError('arms did not evaluate identical data')
            shared_fp = fp
            for row in measured:
                def correct(name):
                    return sum(y==target for y,target in zip(row['predictions'][name],row['target']))
                if correct('reset')!=1 or (row['task']=='acquisition' and correct('no_intervening_writes_oracle')!=1):
                    raise ValueError('paired ceiling violation')
                if len(row['gate_trace_mean_min_max'])!=row['H']+row['D'] or row['state_scalars']!=768:
                    raise ValueError('work/state trace shape mismatch')
            rows[arm] = measured
        if all_eval&set(shared_fp):
            raise ValueError('cross-seed duplicated evaluation')
        all_eval.update(shared_fp)
        replayed.append({'model_seed':seed,'evaluation_seed':eval_seed,'rows':rows})
    expected = summarize(replayed)
    if expected!=report['summary']:
        raise ValueError('summary mismatch')
    return {'status':'REPLAY_VALIDATED_DEVELOPMENT_ONLY','summary':expected}


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('directory',type=Path)
    args = parser.parse_args()
    torch.set_num_threads(2)
    print(json.dumps(validate(args.directory),indent=2))
