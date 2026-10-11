#!/usr/bin/env python3
"""Measure the fixed DEV-004 protocol on immutable DEV-003 checkpoints."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))

import torch  # noqa: E402
from tac_osm.development.memory_diagnostics import (  # noqa: E402
    ID,protocol,train_probe,evaluate,summary,
)
from tac_osm.development.noisy_memory import MemoryConfig,NoisyMemoryModel,parameter_hash  # noqa: E402

SOURCES = (
    'development_protocols/MEMORY-DIAGNOSTICS-DEV-004.json',
    'src/tac_osm/development/memory_diagnostics.py',
    'src/tac_osm/development/noisy_memory.py',
    'src/tac_osm/development/conditional_compute.py',
    'src/tac_osm/development/memory_experiment.py',
    'src/tac_osm/integration_002_benchmark.py',
    'scripts/run_memory_diagnostics_dev.py',
    'scripts/validate_memory_diagnostics_dev.py',
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_parent(directory: Path,seed: int,parent: dict) -> NoisyMemoryModel:
    entries = [c for c in parent['checkpoints'] if c['train_seed']==seed]
    if len(entries)!=1 or parent['experiment_id']!='INTEGRATED-MEMORY-DEV-003' or parent['smoke']:
        raise ValueError('wrong/missing parent measurement')
    entry = entries[0]
    if entry['name'] != f'model-{seed}.pt':
        raise ValueError('wrong parent checkpoint path')
    path = directory/entry['name']
    if digest(path)!=entry['sha256']:
        raise ValueError('parent checkpoint byte hash mismatch')
    payload = torch.load(path,map_location='cpu',weights_only=True)
    model = NoisyMemoryModel(MemoryConfig(**payload['config'])).eval()
    model.load_state_dict(payload['state_dict'],strict=True)
    training = next(t for t in parent['training'] if t['seed']==seed)
    if parameter_hash(model)!=training['final_hash']:
        raise ValueError('parent model identity mismatch')
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--parent',type=Path,required=True)
    parser.add_argument('--output',type=Path,default=ROOT/'artifacts'/ID)
    parser.add_argument('--smoke',action='store_true')
    args = parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2)
    started = time.perf_counter()
    p = protocol()
    parent = json.loads((args.parent/'report.json').read_text())
    # Parent source consumed here must be the same model/generator code whose
    # checkpoint identity was validated in the preceding experiment.
    for name in ('src/tac_osm/development/noisy_memory.py',
                 'src/tac_osm/development/conditional_compute.py',
                 'src/tac_osm/development/memory_experiment.py',
                 'src/tac_osm/integration_002_benchmark.py'):
        if digest(ROOT/name)!=parent['provenance']['sources'][name]:
            raise ValueError('parent source changed')
    shutil.copyfile(args.parent/'report.json',args.output/'parent-report.json')
    records,probes = [],[]
    seed_pairs = list(zip(p['training_seeds'],p['diagnostic_seeds']))[:1 if args.smoke else 3]
    for model_seed,diagnostic_seed in seed_pairs:
        model = load_parent(args.parent,model_seed,parent)
        before = parameter_hash(model)
        probe,training = train_probe(model,model_seed,smoke=args.smoke)
        if not training['nonzero_finite_gradient'] or training['initial_hash']==training['final_hash']:
            raise AssertionError('probe did not learn')
        record = evaluate(model,probe,model_seed,diagnostic_seed,smoke=args.smoke)
        if parameter_hash(model)!=before:
            raise AssertionError('diagnostics changed parent parameters')
        eval_fp = {x for r in record['retrieval'] for x in r['fingerprints']}
        if eval_fp & set(training['train_fingerprints']):
            raise AssertionError('probe training/evaluation overlap')
        source = args.parent/f'model-{model_seed}.pt'
        shutil.copyfile(source,args.output/source.name)
        path = args.output/f'probe-{model_seed}.pt'
        torch.save({'state_dict':probe.state_dict(),'inputs':model.config.experts*model.config.value_dim,
                    'hidden':p['probe']['hidden_dim']},path)
        probes.append({'model_seed':model_seed,'diagnostic_seed':diagnostic_seed,
                       'name':path.name,'sha256':digest(path),'training':training})
        records.append(record)
        print(json.dumps({'model_seed':model_seed,'summary':summary([record])}),flush=True)
    report = {'experiment_id':ID,'status':'DEVELOPMENT_ONLY','confirmatory':False,
              'smoke':args.smoke,'protocol':p,'records':records,'probes':probes,
              'summary':summary(records),'parent_report_sha256':digest(args.output/'parent-report.json'),
              'source_sha256':{name:digest(ROOT/name) for name in SOURCES},
              'provenance':{'commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                            'torch':torch.__version__,'python':sys.version},
              'runtime_seconds':time.perf_counter()-started}
    (args.output/'report.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
    print(json.dumps(report['summary'],indent=2),flush=True)
    print('DEVELOPMENT DIAGNOSTICS ONLY:',args.output,flush=True)


if __name__ == '__main__':
    main()
