"""No-fit preregistered ME077 development scoring and episode uncertainty."""
import json
from pathlib import Path
import numpy as np
from scripts.me077_selective_physical import OUT,ROOT,SEEDS,sha,write


def aggregate(rows, mask):
    chosen=[r for r in rows if mask(r)]
    fields=('new','old','persist','servo','brier','coverage','width','interval_score')
    return {'n':len(chosen),**{k:float(np.mean([r[k] for r in chosen])) for k in fields}} if chosen else {'n':0}


def bootstrap(rows, comparator):
    episodes=sorted({r['episode'] for r in rows})
    totals=np.array([[sum(r['new']-r[comparator] for r in rows if r['episode']==ep),
                      sum(r['episode']==ep for r in rows)] for ep in episodes])
    rng=np.random.default_rng(77077)
    draws=rng.integers(len(episodes),size=(5000,len(episodes)))
    sampled=totals[draws].sum(axis=1)
    values=sampled[:,0]/sampled[:,1]
    return {'method':'paired episode-cluster percentile bootstrap; seed77077;5000resamples; fixed fitted seeds averaged per query',
      'real_episode_units':len(episodes),'upper_lower_95':np.quantile(values,[.025,.975]).tolist(),
      'conditional_on_fitted_models':True,'exposed_DEV':True}


def main():
    raw=[json.loads((OUT/f'seed-{seed}'/'ORIGINAL_PREDICTIONS.json').read_text()) for seed in SEEDS]
    merged=[]
    for group in zip(*raw):
        first=group[0];identity=(first['task'],first['episode_index'],first['frame_index'])
        if any((r['task'],r['episode_index'],r['frame_index'])!=identity for r in group):raise RuntimeError('real identities differ')
        merged.append(dict(task=first['task'],episode=f"{first['task']}:{first['episode_index']}",
          moving=first['motion_rich'],new=float(np.mean([r['physical_mae_model'] for r in group])),
          old=float(np.mean([r['prior_ME076_physical_mae'] for r in group])),persist=first['physical_mae_persistence'],
          servo=first['physical_mae_servo'],brier=float(np.mean([r['motion_brier'] for r in group])),
          coverage=float(np.mean([r['coordinate_interval_coverage'] for r in group])),
          width=float(np.mean([r['physical_interval_width'] for r in group])),
          interval_score=float(np.mean([r['physical_interval_score'] for r in group]))))
    byseed=[]
    for seed,rows in zip(SEEDS,raw):
        fields={k:float(np.mean([r[k] for r in rows])) for k in ('physical_mae_model','physical_mae_persistence','prior_ME076_physical_mae','normalized_mixture_nll')}
        byseed.append({'seed':seed,**fields,'all_case_5pct_gain_vs_both':fields['physical_mae_model']<=.95*min(fields['physical_mae_persistence'],fields['prior_ME076_physical_mae'])})
    allscore=aggregate(merged,lambda r:True);moving=aggregate(merged,lambda r:r['moving']);still=aggregate(merged,lambda r:not r['moving'])
    tasks={task:aggregate(merged,lambda r:r['task']==task) for task in sorted({r['task'] for r in merged})}
    ci={name:bootstrap(merged,name) for name in ('old','persist','servo')}
    gates={'every_seed_5pct_allcase_gain_vs_old_and_persistence':all(r['all_case_5pct_gain_vs_both'] for r in byseed),
      'moving_mean_5pct_gain_vs_persistence':moving['new']<=.95*moving['persist'],
      'no_task_10pct_regression_vs_persistence':all(r['new']<=1.10*r['persist'] for r in tasks.values()),
      'episode_CI_upper_below_zero_vs_persistence':ci['persist']['upper_lower_95'][1]<0}
    outcome={'study':'ME077','real_queries':len(merged),'distinct_real_episodes':len({r['episode'] for r in merged}),
      'all':allscore,'movement_rich':moving,'near_stationary':still,'tasks':tasks,'seeds':byseed,
      'episode_uncertainty':ci,'registered_gates':gates,'registered_gain_gate_passed':all(gates.values()),
      'claim_tier':'DEV_GAIN' if all(gates.values()) else 'MIXED_TRADEOFF' if allscore['new']<allscore['old'] else 'REGRESSION',
      'unseen_generalization':False,'architecture_superiority':False,'customer_value':'NOT_MEASURED',
      'resource_comparison':'new1024updates*24targets vs old160*12; systems comparison, NOT architecture effect',
      'original_summary_sha256':sha(OUT/'ORIGINAL_SUMMARY.json')}
    write(ROOT/'CAPABILITY_RESULTS.json',outcome)
    print(json.dumps(outcome,indent=2))

if __name__=='__main__':main()
