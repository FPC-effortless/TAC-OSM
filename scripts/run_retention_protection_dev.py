#!/usr/bin/env python3
"""Train all predeclared protection arms on identical batches and save evidence."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))

import torch  # noqa: E402
from tac_osm.development.protection_experiment import ID,protocol,train,evaluate,summarize  # noqa: E402

SOURCES = (
    'development_protocols/RETENTION-PROTECTION-DEV-005.json',
    'src/tac_osm/development/protection_benchmark.py',
    'src/tac_osm/development/protected_memory.py',
    'src/tac_osm/development/protection_experiment.py',
    'src/tac_osm/development/noisy_memory.py',
    'src/tac_osm/development/conditional_compute.py',
    'src/tac_osm/development/memory_diagnostics.py',
    'scripts/run_retention_protection_dev.py',
    'scripts/validate_retention_protection_dev.py',
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,default=ROOT/'artifacts'/ID)
    parser.add_argument('--smoke',action='store_true')
    args = parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2)
    p = protocol()
    records,training,checkpoints = [],[],[]
    started = time.perf_counter()
    seeds = list(zip(p['model_seeds'],p['evaluation_seeds']))[:1 if args.smoke else 3]
    for model_seed,eval_seed in seeds:
        models,t = train(model_seed,smoke=args.smoke)
        training.append(t)
        rows = {}
        for arm,model in models.items():
            rows[arm] = evaluate(model,eval_seed,smoke=args.smoke)
            fp = {x for r in rows[arm] for x in r['fingerprints']}
            if fp & set(t['train_fingerprints']):
                raise AssertionError('train/evaluation overlap')
            path = args.output/f'{arm}-{model_seed}.pt'
            torch.save({'config':asdict(model.config),'mode':arm,'state_dict':model.state_dict()},path)
            checkpoints.append({'arm':arm,'model_seed':model_seed,'evaluation_seed':eval_seed,
                                'name':path.name,'sha256':digest(path)})
        records.append({'model_seed':model_seed,'evaluation_seed':eval_seed,'rows':rows})
        print(json.dumps({'seed':model_seed,'summary':summarize([records[-1]])}),flush=True)
    report = {'experiment_id':ID,'status':'DEVELOPMENT_ONLY','confirmatory':False,'smoke':args.smoke,
              'protocol':p,'records':records,'training':training,'checkpoints':checkpoints,
              'summary':summarize(records),'source_sha256':{s:digest(ROOT/s) for s in SOURCES},
              'provenance':{'commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                            'torch':torch.__version__,'python':sys.version},
              'runtime_seconds':time.perf_counter()-started}
    (args.output/'report.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
    print(json.dumps(report['summary'],indent=2),flush=True)
    print('DEVELOPMENT ONLY:',args.output,flush=True)


if __name__=='__main__':
    main()
