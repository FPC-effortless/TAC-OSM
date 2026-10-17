"""Retrospective frozen DEV052B forecast lesions, margins, proper scores and stats."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
import numpy as np
import torch
from tac_osm.development import learned_belief_world as old
from tac_osm.development.research_statistics import paired_summary
from run_learned_belief_dev052b import digest, write

PROTO = ROOT/'research/dev053a/protocol.json'
ORIGINAL_CONTROLLER = old.Controller


def action_values(p, attempts, senses, params, penalty=4.):
    accuracy, rates = params
    values = {"done": penalty*p}
    if attempts < 3:
        values['service'] = 1.2+old.bellman(p*(1-rates[attempts]), attempts+1, senses, params, penalty)[0]
    if senses < 4:
        positive = p*accuracy+(1-p)*(1-accuracy)
        values['inspect'] = .1+positive*old.bellman(old.posterior(p, 1, accuracy), attempts, senses+1, params, penalty)[0]+(1-positive)*old.bellman(old.posterior(p, 0, accuracy), attempts, senses+1, params, penalty)[0]
    ordered = sorted(values.items(), key=lambda z: (z[1], {'inspect': 0, 'service': 1, 'done': 2}[z[0]]))
    return values, ordered[1][1]-ordered[0][1] if len(ordered) > 1 else None


def cross_entropy(p, q):
    p = max(1e-7, min(1-1e-7, p))
    return -q*math.log(p)-(1-q)*math.log(1-p)


def run_intervention(spec, params, model, cfg, intervention, pool=None, smoke=False):
    rows, diagnostic = [], []
    cursor = 0
    for index in range(min(spec['worlds'], 3) if smoke else spec['worlds']):
        _, true_params, _ = old.make_world(spec['seed'], index, spec.get('shifted', False))
        class Recorder(ORIGINAL_CONTROLLER):
            def begin(self, entity, record):
                super().begin(entity, record)
                self.true_belief = getattr(self, 'true_belief', {})
                self.true_belief[entity] = .5

            def observe(self, entity, kind, receipt):
                accepted = super().observe(entity, kind, receipt)
                if accepted and kind == 'inspect':
                    self.true_belief[entity] = old.posterior(self.true_belief[entity], int(receipt['fault']), true_params[0])
                elif accepted and kind == 'service':
                    self.true_belief[entity] *= 1-true_params[1][self.states[entity].attempts-1]
                return accepted

            def forecast(self, entity):
                nonlocal cursor
                p, rates, logits = super().forecast(entity)
                if intervention == 'zeroed_probabilities':
                    return 0., (.85, (0., 0., 0.)), logits
                if intervention == 'shuffled_forecast_rows':
                    vector = pool[cursor % len(pool)]
                    cursor += 1
                    return vector[0], (vector[1], tuple(vector[2:])), logits
                return p, rates, logits

            def choose(self, entity):
                command = super().choose(entity)
                p, params_used = self.predictions[entity]
                s = self.states[entity]
                values, gap = action_values(p, s.attempts, s.senses, params_used, 4*self.calibration['risk'])
                assert abs(values[old.COMMANDS[command]]-min(values.values())) < 1e-6
                truth = self.true_belief[entity]
                predicted = [p]+[p*(1-q) for q in params_used[1]]
                exact = [truth]+[truth*(1-q) for q in true_params[1]]
                # Forecasts of already-used/unavailable attempts have no operational meaning.
                valid = [0]+([s.attempts+1] if s.attempts < 3 else [])
                diagnostic.append({'index': index, 'entity': entity, 'command': old.COMMANDS[command],
                    'attempts': s.attempts, 'senses': s.senses, 'q_values': values, 'top2_gap': gap,
                    'vector': [p, params_used[0], *params_used[1]],
                    'predicted_risks': predicted, 'oracle_conditional_risks': exact,
                    'valid_scoring_indices': valid,
                    'forecast_expected_nll': sum(cross_entropy(predicted[k], exact[k]) for k in valid)/len(valid),
                    'oracle_entropy': sum(cross_entropy(exact[k], exact[k]) for k in valid)/len(valid),
                    'wait_nll': cross_entropy(p, truth)})
                return command
        old.Controller = Recorder
        try:
            row = old.rollout(spec, index, 'joint', params, model, cfg)
        finally:
            old.Controller = ORIGINAL_CONTROLLER
        rows.append(row)
    return {'summary': old.summarize(rows), 'rows': rows, 'decision_diagnostics': diagnostic}


def commit_evidence_hash(evidence):
    index = json.loads((ROOT/'research/dev052b/evidence-index.json').read_text())
    archive = Path(evidence)
    assert digest(archive) == index['archive_sha256']
    commit = index['evidence_commit']
    payload = subprocess.check_output(['git', 'cat-file', 'commit', commit], cwd=ROOT)
    canonical = b'commit '+str(len(payload)).encode()+b'\0'+payload
    assert hashlib.sha1(canonical).hexdigest() == commit
    return {**index, 'commit_payload_sha256': hashlib.sha256(payload).hexdigest(),
            'canonical_commit_object_bytes_sha256': hashlib.sha256(canonical).hexdigest(),
            'hash_definition': 'SHA256(raw git cat-file commit payload); and SHA256(b"commit "+decimal_length+b"\\0"+payload). Git object IDs remain SHA1; these are explicitly defined additional byte digests, not migrated Git SHA256 IDs.'}


def run(evidence, archive, output, smoke=False):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    proto = json.loads(PROTO.read_text())
    original = Path(evidence)
    out = Path(output); out.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    common = json.loads((original/'common.json').read_text())
    original_proto = json.loads((ROOT/'research/dev052b/fresh-execution-protocol.json').read_text())
    assert common['protocol_sha256'] == digest(ROOT/'research/dev052b/fresh-execution-protocol.json')
    data = json.loads((original/'training-data.json').read_text())
    provenance = commit_evidence_hash(archive)
    seeds = proto['seeds'][:1] if smoke else proto['seeds']
    metadata = {'study': 'DEV-053A', 'retrospective': True, 'smoke': smoke,
                'source': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                'protocol_sha256': digest(PROTO), 'evidence_provenance': provenance,
                'experiments': {}, 'old_arm_crossed_statistics': {}}
    for seed in seeds:
        report = json.loads((original/f'seed{seed}/report.json').read_text())
        model = old.BeliefModel(seed, 'joint')
        checkpoint = original/f'seed{seed}/joint.pt'
        assert digest(checkpoint) == report['arms']['joint']['checkpoint_sha256']
        model.load_state_dict(torch.load(checkpoint, weights_only=True)); model.eval()
        metadata['experiments'][str(seed)] = {}
        for cohort in ('in_regime', 'shifted'):
            spec = {**original_proto['splits'][cohort], 'shifted': cohort == 'shifted'}
            cfg = report['arms']['joint']['calibration']
            intact = run_intervention(spec, data['estimated_parameters'], model, cfg, 'intact', smoke=smoke)
            expected = report['arms']['joint']['evaluation'][cohort]['rows']
            assert old.strip_timing(intact['rows']) == old.strip_timing(expected[:len(intact['rows'])])
            pool = [row['vector'] for row in intact['decision_diagnostics']]
            random.Random(f'dev053a-shuffle:{seed}:{cohort}').shuffle(pool)
            results = {'intact': intact}
            for intervention in proto['interventions'][1:]:
                print(f'{seed} {cohort} {intervention}', flush=True)
                results[intervention] = run_intervention(spec, data['estimated_parameters'], model, cfg, intervention, pool, smoke)
            path = out/f'seed{seed}-{cohort}.json'
            write(path, results)
            summary = {}
            for name, result in results.items():
                rows = result['decision_diagnostics']
                gaps = [r['top2_gap'] for r in rows if r['top2_gap'] is not None]
                summary[name] = {'native_summary': result['summary'],
                    'mean_expected_nll': float(np.mean([r['forecast_expected_nll'] for r in rows])),
                    'mean_oracle_entropy': float(np.mean([r['oracle_entropy'] for r in rows])),
                    'mean_excess_nll': float(np.mean([r['forecast_expected_nll']-r['oracle_entropy'] for r in rows])),
                    'top2_gap_quantiles_0_25_50_75_100': np.quantile(gaps, [0, .25, .5, .75, 1]).tolist(),
                    'fraction_gaps_over_0_1': float(np.mean(np.array(gaps) > .1)),
                    'native_actions_changed_worlds': sum(a['calls'] != b['calls'] for a, b in zip(result['rows'], intact['rows'])),
                    'native_cost_changed_worlds': sum(a['cost'] != b['cost'] for a, b in zip(result['rows'], intact['rows']))}
            metadata['experiments'][str(seed)][cohort] = {'artifact': path.name, 'sha256': digest(path), 'summary': summary}
    for cohort in ('in_regime', 'shifted'):
        baseline = common['baselines']['estimated_bayes'][cohort]['rows']
        oracle = common['baselines']['known_bayes'][cohort]['rows']
        strata = [r['index'] % 4 for r in baseline] if cohort == 'shifted' else None
        metadata['old_arm_crossed_statistics'][cohort] = {}
        for arm in old.NEURAL:
            reports = [json.loads((original/f'seed{s}/report.json').read_text()) for s in seeds]
            costs = [[r['cost']/2 for r in report['arms'][arm]['evaluation'][cohort]['rows']] for report in reports]
            metadata['old_arm_crossed_statistics'][cohort][arm] = paired_summary(costs, [r['cost']/2 for r in baseline], [r['cost']/2 for r in oracle], strata)
    metadata['seconds'] = time.monotonic()-start
    metadata['status'] = 'COMPLETED_RETROSPECTIVE_DIAGNOSTICS' if not smoke else 'SMOKE_ONLY'
    write(out/'review.json', metadata)
    print(json.dumps({'status': metadata['status'], 'seconds': metadata['seconds']}), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--evidence', required=True); p.add_argument('--archive', required=True)
    p.add_argument('--output', required=True); p.add_argument('--smoke', action='store_true')
    a = p.parse_args(); run(a.evidence, a.archive, a.output, a.smoke)
