#!/usr/bin/env python3
"""Guarded SENSE-TOK worker launch. Default: preflight only, never fit.

Execution requires an independently signed execution-bundle digest, canonical
training parity, closed external math/probe/source gates and protected dual-T4
approval. Final qualification still uses run_qualified_research.py and the
access-separated profiler/evaluator; workers cannot sign or promote themselves.
"""
import argparse
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT)]
from scripts.validate_research_parity import check_manifest
from tac_osm.adaptive_token_controls import ARM_NAMES, TokenComparisonArm
from tac_osm.adaptive_token_training import load_corpus, digest, fit_worker, sample_order_digest

REQUIRED = ('qualified_protocol', 'corpus_contract', 'full_graph_math_and_probe_receipt')


def preflight(manifest_path, evidence_path, bundle_path, source_root):
    manifest = json.loads(manifest_path.read_bytes())
    if manifest.get('study_type') != 'architecture_comparison' or manifest.get('study_id') != 'SENSE-TOK-006-LEARNING':
        raise ValueError('reviewed SENSE-TOK training registration required; no-fit preparation cannot launch')
    evidence = json.loads(evidence_path.read_bytes())
    result = check_manifest(manifest, 'preflight', reward_attestation=evidence,
                           trusted_reward_public_key_hex=os.environ.get('TAC_OSM_REWARD_VERIFIER_PUBLIC_KEY_HEX'))
    raw = bundle_path.read_bytes()
    if digest(raw) != evidence['payload']['protected_assets_sha256']:
        raise ValueError('execution bundle not bound to independent signed protected-assets digest')
    bundle = json.loads(raw)
    if set(bundle) != set(REQUIRED):
        raise ValueError('exact execution bundle required')
    documents = {}
    for name, record in bundle.items():
        path = Path(record['path'])
        if not path.is_absolute() or path.is_symlink():
            raise ValueError('trusted absolute artifact paths required')
        data = path.read_bytes()
        if digest(data) != record['sha256']:
            raise ValueError('protected artifact changed')
        documents[name] = json.loads(data)
    config = documents['qualified_protocol']
    if config.get('training_authorized') is not True or config.get('no_new_model_fits') is not False:
        raise ValueError('protocol still prohibits training')
    receipt = documents['full_graph_math_and_probe_receipt']
    if receipt.get('study_id') != manifest['study_id'] or any(receipt.get(k) != 'PASS' for k in
        ('full_graph_source_correspondence', 'actual_source_probe', 'sealed_test_custody', 'source_qualification', 'full_flop_prefit_profile', 'near_duplicate_audit')):
        raise ValueError('independent full-graph/source/probe/custody/resource qualification missing')
    expected = {(arm, seed) for arm in ARM_NAMES for seed in (1,2,3,4,5)}
    rows = manifest['arms']
    if len(rows) != len(expected) or {(r['architecture'], r['seed']) for r in rows} != expected:
        raise ValueError('complete six-arm/five-seed matrix required')
    for row in rows:
        count = TokenComparisonArm(row['architecture'], seed=row['seed'],
            random_byte_probability=config['random_byte_probability'] if row['architecture']=='random_gate' else None).parameter_receipt()
        if any(count[k] != row[k] for k in ('trainable_parameters','deployed_parameters')):
            raise ValueError('live model counts differ from registration')
        b = row['budget']
        if (b['steps'],b['batch_size'],b['history_events']) != (config['steps_per_arm_seed'],config['batch_size'],config['history_bytes']):
            raise ValueError('worker exposure differs from frozen registration')
    corpus_raw = documents['corpus_contract']
    if digest(json.dumps(corpus_raw,sort_keys=True,separators=(',',':')).encode()) != manifest['real_world_source']['sha256']:
        raise ValueError('corpus source contract differs from signed registration')
    corpus = load_corpus(source_root, corpus_raw, config['history_bytes'])
    for row in rows:
        actual = sample_order_digest(corpus['FIT'], seed=row['seed'],
            steps=config['steps_per_arm_seed'], batch_size=config['batch_size'])
        if actual != row['training_sample_order_sha256']:
            raise ValueError('actual sample order differs from registration')
    return config, corpus, result


def shard(index, config, corpus, output):
    import torch
    torch.cuda.set_device(index)
    for seed in (1,2,3,4,5):
        if (seed-1)%2 != index:
            continue
        for arm in ARM_NAMES:
            fit_worker(config, corpus, arm=arm, seed=seed, device=f'cuda:{index}',
                       output=output/f'{arm}-seed-{seed}',
                       random_probability=config['random_byte_probability'] if arm=='random_gate' else None)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    for field in ('manifest','evidence','bundle','source-root','output'):
        p.add_argument('--'+field, type=Path, required=True)
    p.add_argument('--execute', action='store_true')
    args=p.parse_args(argv)
    try:
        if args.output.resolve().is_relative_to(args.source_root.resolve()):
            raise ValueError('output must be outside the protected source root')
        config, corpus, result = preflight(args.manifest,args.evidence,args.bundle,args.source_root)
        if not args.execute:
            print(json.dumps({'status':'PREFLIGHT_ONLY_NO_FIT','audit':result}))
            return 0
        if os.environ.get('PHONE_LAB_GPU_EXECUTION_ENABLED') != 'true':
            raise ValueError('protected operator GPU approval missing')
        import torch
        if torch.cuda.device_count()!=2 or any('T4' not in torch.cuda.get_device_name(i) for i in (0,1)):
            raise ValueError('exactly two physical T4 devices required')
        for i in (0,1):
            if float((torch.ones(1, device=f'cuda:{i}')*2).cpu()) != 2:
                raise ValueError('physical device operation failed')
        args.output.mkdir(parents=True, exist_ok=False)
        # Spawned workers receive no signing, GitHub or evaluator credentials.
        allowed={'PATH','PYTHONPATH','HOME','LANG','TMPDIR','CUDA_VISIBLE_DEVICES','OMP_NUM_THREADS','MKL_NUM_THREADS'}
        previous=dict(os.environ)
        context=multiprocessing.get_context('spawn')
        workers=[]
        try:
            for key in list(os.environ):
                if key not in allowed:
                    del os.environ[key]
            for index in (0,1):
                worker=context.Process(target=shard,args=(index,config,corpus,args.output))
                worker.start()
                workers.append(worker)
            for worker in workers:
                worker.join()
            if any(worker.exitcode!=0 for worker in workers):
                raise ValueError('seed shard failed; no qualification')
        finally:
            for worker in workers:
                if worker.is_alive():
                    worker.terminate()
                    worker.join()
            os.environ.clear()
            os.environ.update(previous)
        print(json.dumps({'status':'WORKERS_FINISHED_UNQUALIFIED','final_external_audit_required':True}))
        return 0
    except (ValueError, KeyError, OSError, RuntimeError) as exc:
        print(json.dumps({'status':'BLOCKED_BEFORE_OR_DURING_FIT','reason':str(exc),'qualified':False}))
        return 2


if __name__=='__main__':
    raise SystemExit(main())
