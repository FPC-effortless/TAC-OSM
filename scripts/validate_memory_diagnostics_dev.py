#!/usr/bin/env python3
"""Reload frozen models/probes and replay every DEV-004 diagnostic row."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))

import torch  # noqa: E402
from tac_osm.development.memory_diagnostics import (  # noqa: E402
    ID,ReadProbe,protocol,evaluate,summary,
)
from tac_osm.development.noisy_memory import parameter_hash  # noqa: E402
from run_memory_diagnostics_dev import SOURCES,digest,load_parent  # noqa: E402


def validate(directory: Path) -> dict:
    report = json.loads((directory/'report.json').read_text())
    p = protocol()
    if report['experiment_id']!=ID or report['status']!='DEVELOPMENT_ONLY' or report['confirmatory'] is not False:
        raise ValueError('wrong identity or promoted claim')
    if report['protocol']!=p or set(report['source_sha256'])!=set(SOURCES):
        raise ValueError('changed/incomplete protocol or source provenance')
    for name,expected in report['source_sha256'].items():
        if digest(ROOT/name)!=expected:
            raise ValueError('source hash mismatch')
    if digest(directory/'parent-report.json')!=report['parent_report_sha256']:
        raise ValueError('parent report mismatch')
    parent = json.loads((directory/'parent-report.json').read_text())
    seeds = list(zip(p['training_seeds'],p['diagnostic_seeds']))[:1 if report['smoke'] else 3]
    if len(report['records'])!=len(seeds) or len(report['probes'])!=len(seeds):
        raise ValueError('missing/extra model seeds')
    replayed,all_eval = [],set()
    for (model_seed,diagnostic_seed),probe_record,stored in zip(seeds,report['probes'],report['records']):
        if (probe_record['model_seed'],probe_record['diagnostic_seed'])!=(model_seed,diagnostic_seed):
            raise ValueError('wrong seed pair')
        model = load_parent(directory,model_seed,parent)
        if probe_record['name']!=f'probe-{model_seed}.pt':
            raise ValueError('wrong probe path')
        path = directory/probe_record['name']
        if digest(path)!=probe_record['sha256']:
            raise ValueError('probe byte hash mismatch')
        payload = torch.load(path,map_location='cpu',weights_only=True)
        if payload['inputs']!=model.config.experts*model.config.value_dim or payload['hidden']!=p['probe']['hidden_dim']:
            raise ValueError('changed probe architecture')
        probe = ReadProbe(payload['inputs'],payload['hidden']).eval()
        probe.load_state_dict(payload['state_dict'],strict=True)
        t = probe_record['training']
        if (t['model_seed']!=model_seed or t['steps']!=(2 if report['smoke'] else p['probe']['steps'])
                or t['pairs']!=(8 if report['smoke'] else p['probe']['train_pairs'])
                or t['final_hash']!=parameter_hash(probe) or t['parent_hash']!=parameter_hash(model)
                or t['initial_hash']==t['final_hash'] or not t['nonzero_finite_gradient']):
            raise ValueError('probe training/checkpoint identity mismatch')
        rows = evaluate(model,probe,model_seed,diagnostic_seed,smoke=report['smoke'])
        if json.loads(json.dumps(rows))!=stored:
            raise ValueError('raw diagnostic predictions/state/carry/calibration differ from replay')
        fingerprints = [x for r in rows['retrieval'] for x in r['fingerprints']]
        if len(fingerprints)!=len(set(fingerprints)) or all_eval & set(fingerprints) or set(t['train_fingerprints']) & set(fingerprints):
            raise ValueError('duplicate/overlapping evaluation cases')
        all_eval.update(fingerprints)
        for carry in rows['carry']:
            for name in ('no_commit','reset'):
                if sum(y==t for y,t in zip(carry['predictions'][name],carry['target']))!=1:
                    raise ValueError('paired carry ceiling violated')
        replayed.append(rows)
    expected = summary(replayed)
    if expected!=report['summary']:
        raise ValueError('summary mismatch')
    return {'status':'REPLAY_VALIDATED_DEVELOPMENT_ONLY','summary':expected}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('directory',type=Path)
    args = parser.parse_args()
    torch.set_num_threads(2)
    print(json.dumps(validate(args.directory),indent=2))
