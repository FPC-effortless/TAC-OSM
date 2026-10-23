"""No-fit original ME078 all-horizon metrics and preregistered paired uncertainty."""
import json
from pathlib import Path
import numpy as np
from scripts.me078_train_world import ROOT,OUT,ARMS,SEEDS,sha,write,HORIZONS


def ranking(rows):
    y=np.array([r['moving'] for r in rows],bool);p=np.array([r['motion_probability'] for r in rows]);npos=int(y.sum());nneg=len(y)-npos
    if not npos or not nneg:return {'positive':npos,'negative':nneg,'auc':None,'ap':None}
    scores=np.unique(p)[::-1];tp=fp=0;ap=0.
    for score in scores:
        mask=p==score;pos=int(y[mask].sum());neg=int(mask.sum())-pos
        tp+=pos;fp+=neg;ap+=pos/npos*tp/(tp+fp)
    auc=float(((p[y,None]>p[~y][None,:]).mean())+.5*((p[y,None]==p[~y][None,:]).mean()))
    top=np.argsort(-p,kind='stable')[:max(1,int(np.ceil(.1*len(y))))]
    thresholds={}
    for threshold in (.1,.5):
        active=p>=threshold;true=int((active&y).sum());false=int((active&~y).sum())
        thresholds[str(threshold)]={'alerts':int(active.sum()),'true_positive':true,'false_positive':false,'precision':true/max(1,int(active.sum())),'recall':true/npos}
    return {'positive':npos,'negative':nneg,'auc':auc,'ap':float(ap),'top10pct_motion_capture':float(y[top].sum()/npos),'fixed_thresholds':thresholds}


def summary(rows):
    if not rows:return {'n':0}
    fields=('mae','persistence_mae','servo_mae','velocity_mae','motion_brier','coverage','width','interval_score')
    return {'n':len(rows),**{k:float(np.mean([r[k] for r in rows])) for k in fields},'event_ranking':ranking(rows)}


def main():
    raw={}
    for arm in ARMS:
        for seed in SEEDS:
            path=OUT/arm/f'seed-{seed}'/'ORIGINAL_PREDICTIONS.json';raw[(arm,seed)]=json.loads(path.read_text())
    identities=lambda rows:[(r['task'],r['episode'],r['frame'],r['horizon']) for r in rows]
    expected=identities(raw[(ARMS[0],SEEDS[0])])
    if any(identities(rows)!=expected for rows in raw.values()):raise RuntimeError('architecture/seed case IDs differ')
    if any([r['target'] for r in rows]!=[r['target'] for r in raw[(ARMS[0],SEEDS[0])]] for rows in raw.values()):raise RuntimeError('actual target differs')
    merged={}
    for arm in ARMS:
        selected=[]
        for group in zip(*[raw[(arm,s)] for s in SEEDS]):
            r=dict(group[0]);
            for k in ('mae','motion_brier','motion_probability','coverage','width','interval_score'):
                r[k]=float(np.mean([v[k] for v in group]))
            selected.append(r)
        merged[arm]=selected
    metrics={arm:{str(h):{'all':summary([r for r in rows if r['horizon']==h]),
                'moving':summary([r for r in rows if r['horizon']==h and r['moving']]),
                'stationary':summary([r for r in rows if r['horizon']==h and not r['moving']])} for h in HORIZONS}
             for arm,rows in merged.items()}
    perseed={arm:[{'seed':s,'horizons':{str(h):summary([r for r in raw[(arm,s)] if r['horizon']==h]) for h in HORIZONS}} for s in SEEDS] for arm in ARMS}
    def relative_score(means):return float(np.mean([means[str(h)]['mae']/metrics['native'][str(h)]['all']['persistence_mae'] for h in HORIZONS]))
    aggregate={arm:relative_score({h:d['all'] for h,d in byh.items()}) for arm,byh in metrics.items()}
    seedaggregate={arm:[relative_score(r['horizons']) for r in records] for arm,records in perseed.items()}
    # Matched episode resampling; H4/H16 have their own explicit eligible denominators.
    episodes=sorted({(r['task'],r['episode']) for r in merged['native']});n=len(episodes)
    draws=np.random.default_rng(78078).integers(n,size=(5000,n))
    totals={}
    for arm in ARMS:
        totals[arm]=np.array([[[sum(r['mae'] for r in merged[arm] if (r['task'],r['episode'])==ep and r['horizon']==h),
                                sum((r['task'],r['episode'])==ep and r['horizon']==h for r in merged[arm])] for h in HORIZONS] for ep in episodes])
    sampled={}
    for arm in ARMS:
        t=totals[arm][draws].sum(1)
        means=t[:,:,0]/t[:,:,1]
        baselines=np.array([metrics['native'][str(h)]['all']['persistence_mae'] for h in HORIZONS])
        sampled[arm]=(means/baselines).mean(1)
    cis={arm:np.quantile(sampled['native']-sampled[arm],[.0125,.9875]).tolist() for arm in ('gru','transformer')}
    wins={arm:sum(a<b for a,b in zip(seedaggregate['native'],seedaggregate[arm])) for arm in ('gru','transformer')}
    # Frozen old complete model, H1 same exact original queries; no new fit.
    old=[]
    for seed in SEEDS:
        path=Path('research/replay-bundles/ME077')/f'seed-{77000+seed%1000}'/'ORIGINAL_PREDICTIONS.json'
        values=json.loads(path.read_text());current=[r for r in raw[('native',seed)] if r['horizon']==1]
        if len(values)!=len(current):raise RuntimeError('old/new H1 cohort differs')
        for a,b in zip(values,current):
            if (a['task'],a['episode_index'],a['frame_index'])!=(b['task'],b['episode'],b['frame']) or a['target_normalized']!=b['target']:
                raise RuntimeError('old/new task/target mismatch')
        old.append(float(np.mean([r['physical_mae_model'] for r in values])))
    h1=metrics['native']['1']['all'];moving=metrics['native']['1']['moving']
    tasks={task:summary([r for r in merged['native'] if r['horizon']==1 and r['task']==task]) for task in sorted({r['task'] for r in merged['native']})}
    utility={'H1_5pct_gain_vs_persistence_and_ME077':h1['mae']<=.95*min(h1['persistence_mae'],float(np.mean(old))),
      'H1_moving_5pct_gain_vs_best_physical_reference':moving['mae']<=.95*min(moving[k] for k in ('persistence_mae','servo_mae','velocity_mae')),
      'no_task_10pct_regression_vs_persistence':all(r['mae']<=1.1*r['persistence_mae'] for r in tasks.values())}
    comparison={arm:{'aggregate_2pct_relative_gain':aggregate['native']<=.98*aggregate[arm],'adjusted_episode_CI_upper_below_zero':cis[arm][1]<0,'at_least4of5_seed_wins':wins[arm]>=4} for arm in ('gru','transformer')}
    full=all(utility.values()) and all(v for r in comparison.values() for v in r.values())
    write(ROOT/'CAPABILITY_RESULTS.json',{'study':'ME078','native_registered_full_gain_gate':full,'utility_gates':utility,'architecture_comparison_gates':comparison,
      'horizon_scores':metrics,'all_seed_scores':perseed,'aggregate_horizon_relative_to_persistence':aggregate,'aggregate_each_seed':seedaggregate,
      'paired_seed_wins':wins,'paired_episode_adjusted97_5_CI_native_minus_control':cis,'bootstrap_episode_units':n,'bootstrap_repetitions':5000,
      'bootstrap_convention':'fixed original horizon-persistence mean normalization; five fitted seeds averaged per actual query; paired episode percentile intervals, conditional on fitted models',
      'old_ME077_H1_MAE':float(np.mean(old)),'H1_native_task_scores':tasks,'source_exposure':'EXPOSED_DEV','independent_gain':False,
      'compute_claim_scope':'Matched audited algorithmic scalar-operation estimates only, not a hardware FLOP/instruction equivalence or foundation-scale competition.',
      'claim_tier':'DEV_GAIN' if full else 'MIXED_TRADEOFF' if h1['mae']<np.mean(old) else 'REGRESSION',
      'customer_value':'NOT_MEASURED'})
    print(json.dumps({'all_horizon_MAE':{a:{h:r['all']['mae'] for h,r in x.items()} for a,x in metrics.items()},'native_H1_moving':moving,'utility':utility,'comparison':comparison,'aggregate':aggregate,'CIs':cis,'wins':wins,'full_gate':full},indent=2),flush=True)

if __name__=='__main__':main()
