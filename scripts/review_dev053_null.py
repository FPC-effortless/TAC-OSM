"""Retrospective DEV-053 checks; never rewrites original acceptance rules."""
import json
import math
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
import numpy as np
from tac_osm.development.research_statistics import crossed_draws
from tac_osm.development.joint_repair import calibration_bins
from audit_adaptive_repair_dev053b import matrix
from run_learned_belief_dev052b import write


def review():
    directory = ROOT/'results/dev053b'
    common = json.loads((directory/'common.json').read_text())
    reports = [json.loads(p.read_text()) for p in sorted(directory.glob('seed*/report.json'))]
    def values(arm, family):
        return np.array([matrix(r['arms'][arm]['evaluation'][family]) for r in reports]) if arm in reports[0]['arms'] else matrix(common['baselines'][arm][family])[None]
    strata = [0]*128+[1]*128
    w, ids = crossed_draws(5, 256, strata, replicates=20000, seed=5540001)
    contrasts = {}
    for arm in ('gru128', 'hmm_mixture_dp', 'mtsk_frozen_kernel', 'mtsk_reset'):
        diff = np.concatenate([(values('mtsk', f)-values(arm, f)).mean(-1) for f in ('markov', 'drift')], 1)
        draws = np.take_along_axis(w@diff, ids, 1).mean(1)
        contrasts[arm] = {'delta': float(diff.mean()), 'unadjusted_95pct': np.quantile(draws, [.025, .975]).tolist(),
                          'bonferroni_four_guard_95pct': np.quantile(draws, [.05/8, 1-.05/8]).tolist()}
        if arm != 'hmm_mixture_dp': contrasts[arm]['bonferroni_three_memory_95pct'] = np.quantile(draws, [.05/6, 1-.05/6]).tolist()
    economics = {}
    for family in ('markov', 'drift', 'pooled'):
        families = ('markov', 'drift') if family == 'pooled' else (family,)
        b = np.concatenate([values('stationary_beta', f)[0] for f in families])
        o = np.concatenate([values('oracle', f)[0] for f in families])
        economics[family] = {}
        for i, n in enumerate((0, 4, 8, 16, 32)):
            gap = float((b[:, i]-o[:, i]).mean()); acquisition = .6*n
            row = {'acquisition_per_entity': acquisition, 'oracle_headroom': gap,
                   'optimistic_oracle_break_even_decisions': math.ceil(acquisition/gap) if gap > 0 else None,
                   'models': {}}
            for arm in ('mtsk', 'gru128', 'hmm_mixture_dp'):
                m = np.concatenate([values(arm, f) for f in families], 1)[:, :, i]
                saving = float(b[:, i].mean()-m.mean())
                row['models'][arm] = {'absolute_cost_minus_beta': -saving,
                                     'hypothetical_break_even_decisions': math.ceil(acquisition/saving) if saving > 0 else None}
            economics[family][str(n)] = row
    calibration = {}
    for family in ('in_regime', 'shifted'):
        ps, qs = [], []
        for p in sorted((ROOT/'results/dev053a').glob(f'seed*-{family}.json')):
            for d in json.loads(p.read_text())['intact']['decision_diagnostics']:
                for index in d['valid_scoring_indices']:
                    ps.append(d['predicted_risks'][index]); qs.append(d['oracle_conditional_risks'][index])
        calibration[family] = calibration_bins(ps, qs)
    result = {'study': 'DEV-054 retrospective DEV-053 review', 'interpretation': 'Null on adaptation; reset not a robust multiplicity-adjusted advantage',
              'contrasts': contrasts, 'economics': economics, 'dev053a_calibration': calibration,
              'reset': {'retrained': True, 'updates_per_seed': 256, 'seeds': [r['seed'] for r in reports],
                        'source': 'AdaptiveModel.observe resets state during sequence used by fit_model; separate reset checkpoints reconstructed by prior audit'},
              'training': {'updates_per_fit': 256, 'matrix_total_updates': 6400, 'logged_updates': [1, 256], 'plateau_established': False,
                           'range': 'Train and test share endpoint convex range; old train Markov/triangle, test includes unseen sine, limited one-dimensional rate manifold'},
              'scope': 'MTSK only; no CDL/CASM/selective-computation evidence',
              'ratio_policy': 'Current review suppresses drift32 ratio entirely; original frozen protocol/raw evidence retained unchanged. Future DEV054 floor=.05 and <1% invalid denominator draws.'}
    write(ROOT/'research/dev054/dev053-null-review.json', result)
    print(json.dumps({'reset_adjusted': contrasts['mtsk_reset'], 'pooled_budget32': economics['pooled']['32'], 'calibration': {k: {x: v[x] for x in ('expected_ece','nll','excess_nll')} for k,v in calibration.items()}}, indent=2))


if __name__ == '__main__': review()
