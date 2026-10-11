#!/usr/bin/env python3
"""Train a fixed-budget development pilot; preserve checkpoints and raw traces."""
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
sys.path.insert(0, str(ROOT/'src'))

import torch  # noqa: E402
from tac_osm.development.memory_experiment import (  # noqa: E402
    EXPERIMENT_ID, TRAIN_SEEDS, EVAL_SEEDS, TRAIN_STEPS, EVAL_PAIRS,
    evaluate, summarize, train_model,
)

SOURCES = (
    'src/tac_osm/development/conditional_compute.py',
    'src/tac_osm/development/noisy_memory.py',
    'src/tac_osm/development/memory_experiment.py',
    'src/tac_osm/integration_002_benchmark.py',
    'src/tac_osm/integration_002_learned_dev.py',
    'scripts/run_integrated_memory_dev.py',
    'scripts/validate_integrated_memory_dev.py',
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=ROOT/'artifacts'/EXPERIMENT_ID)
    parser.add_argument('--smoke', action='store_true', help='two steps, one seed; not a scientific measurement')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    started = time.perf_counter()
    steps, pairs = (2, 1) if args.smoke else (TRAIN_STEPS, EVAL_PAIRS)
    seed_pairs = list(zip(TRAIN_SEEDS, EVAL_SEEDS))[:1 if args.smoke else 3]
    trained, rows, checkpoints, per_seed = [], [], [], []
    training_fingerprints, evaluation_fingerprints = set(), set()
    for train_seed, eval_seed in seed_pairs:
        model, record = train_model(train_seed, steps=steps)
        if record['initial_hash'] == record['final_hash'] or not all(record['nonzero_finite_task_gradients'].values()):
            raise AssertionError('missing real parameter/gradient learning')
        if not record['future_decision_write_gradient']:
            raise AssertionError('future decision did not train the post-action write')
        training_fingerprints.update(record['train_fingerprints'])
        seed_rows = evaluate(model, eval_seed, pairs=pairs)
        evaluation_fingerprints.update(x for r in seed_rows for x in r['fingerprints'])
        checkpoint = args.output/f'model-{train_seed}.pt'
        torch.save({'config':asdict(model.config), 'state_dict':model.state_dict()}, checkpoint)
        checkpoints.append({'name':checkpoint.name, 'sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                            'train_seed':train_seed, 'eval_seed':eval_seed,
                            'parameters':sum(p.numel() for p in model.parameters())})
        for r in seed_rows:
            r['train_seed'], r['eval_seed'] = train_seed, eval_seed
        rows.extend(seed_rows)
        trained.append(record)
        per_seed.append({'train_seed':train_seed, 'eval_seed':eval_seed, 'summary':summarize(seed_rows)})
        print(json.dumps(per_seed[-1]), flush=True)
    if training_fingerprints & evaluation_fingerprints:
        raise AssertionError('train/evaluation overlap')
    commit = subprocess.check_output(['git','rev-parse','HEAD'], cwd=ROOT, text=True).strip()
    report = {
        'experiment_id':EXPERIMENT_ID, 'status':'DEVELOPMENT_ONLY', 'smoke':args.smoke,
        'confirmatory':False, 'train_steps':steps, 'eval_pairs':pairs,
        'training':trained, 'checkpoints':checkpoints, 'rows':rows,
        'summary':summarize(rows), 'per_seed':per_seed,
        'provenance':{'commit':commit, 'torch':torch.__version__, 'python':sys.version,
                      'sources':{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in SOURCES}},
        'runtime_seconds':time.perf_counter()-started,
        'limitations':[
            'Development generator influenced design; independent seeds are not untouched confirmatory families.',
            'Adapted gated delta rule, not a faithful implementation of SwiLA or Gated DeltaNet-2.',
            'Verifier learns synthetic sensor-validity evidence, not general reasoning correctness.',
            'Binary answers are not target retrieval recall; learned fixed-state model exposes no slot recall.',
            'Follow-up write test has a short delay and no new intervening writes; not long-term consolidation evidence.',
            'Trace counts are component calls/state elements, not hardware FLOPs or cost-normalized comparisons.',
            'Memory ingestion is O(H+D); fixed-state reads do not establish sublinear total work.',
            'No matched modern recurrent/attention baseline, seed-level confidence interval or SOTA claim.',
            'Full streaming multimodal successor is mechanism-tested separately, not trained by this pilot.',
        ],
    }
    path = args.output/'report.json'
    path.write_text(json.dumps(report, indent=2, sort_keys=True)+'\n')
    print(json.dumps(report['summary'], indent=2))
    print(f'DEVELOPMENT ONLY; checkpoint and raw evidence: {path}', flush=True)


if __name__ == '__main__':
    main()
