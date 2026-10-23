"""Read-only original-score summaries for the fixed ME080 registration."""
import json
from pathlib import Path
import numpy as np
from scripts.me080_train_world import ROOT,OUT,SEEDS,HORIZONS,sha,write
from scripts.me079_report_capability import ranking


def load(path):return json.loads(Path(path).read_text())
def identity(row):return row['task'],row['episode'],row['frame'],row['horizon']


def summarize(rows):
    fields=('mae','persistence_mae','velocity_mae','servo_mae','motion_brier','coverage','width','interval_score')
    if not rows:return {'n':0}
    result={'n':len(rows),**{k:float(np.mean([r[k] for r in rows])) for k in fields},'event_ranking':ranking(rows)}
    if 'physical_marginal_nll' in rows[0]:result['physical_marginal_nll']=float(np.mean([r['physical_marginal_nll'] for r in rows]))
    return result


def main():
    raw=[load(OUT/f'seed-{s}'/'ORIGINAL_PREDICTIONS.json') for s in SEEDS]
    prior=[load(Path('research/replay-bundles/ME079/native')/f'seed-{79001+i}'/'ORIGINAL_PREDICTIONS.json') for i in range(5)]
    older=[load(Path('research/replay-bundles/ME078/native')/f'seed-{78001+i}'/'ORIGINAL_PREDICTIONS.json') for i in range(5)]
    specialist=[load(Path('research/replay-bundles/ME077')/f'seed-{77001+i}'/'ORIGINAL_PREDICTIONS.json') for i in range(5)]
    ids=[identity(r) for r in raw[0]]
    for rows in raw+prior+older:
        if [identity(r) for r in rows]!=ids:raise RuntimeError('Old/new actual query identity differs')
        if any(r['target']!=v['target'] for r,v in zip(rows,raw[0])):raise RuntimeError('Old/new actual target differs')
    one=[r for r in raw[0] if r['horizon']==1]
    for rows in specialist:
        if len(rows)!=len(one):raise RuntimeError('ME077 denominator differs')
        for a,b in zip(rows,one):
            if (a['task'],a['episode_index'],a['frame_index'])!=identity(b)[:3] or a['target_normalized']!=b['target']:
                raise RuntimeError('ME077 original query/target differs')
    def averaged(rows):
        merged=[]
        for group in zip(*rows):
            r=dict(group[0])
            for key in ('mae','motion_brier','coverage','width','interval_score'):
                r[key]=float(np.mean([v[key] for v in group]))
            if 'physical_marginal_nll' in r:r['physical_marginal_nll']=float(np.mean([v['physical_marginal_nll'] for v in group]))
            merged.append(r)
        return merged
    # No averaged probabilities masquerading as one deployed model. Rank each
    # individual original, and only then average its AP/AUC.
    def metrics(rows):
        result={}
        for h in HORIZONS:
            by={}
            for group,select in [('all',lambda r:True),('moving',lambda r:r['moving']),('stationary',lambda r:not r['moving'])]:
                indiv=[summarize([r for r in rr if r['horizon']==h and select(r)]) for rr in rows]
                mean=dict(indiv[0])
                for key in indiv[0]:
                    if isinstance(indiv[0][key],float):mean[key]=float(np.mean([s[key] for s in indiv]))
                mean['event_ranking']={'aggregation':'mean of individual original model metrics; no multi-model deployment',
                  **{k:float(np.mean([s['event_ranking'][k] for s in indiv])) for k in ('ap','auc') if indiv[0]['event_ranking'].get(k) is not None},
                  'positive':indiv[0]['event_ranking']['positive'],'negative':indiv[0]['event_ranking']['negative']}
                by[group]=mean
            result[str(h)]=by
        return result
    newmetrics=metrics(raw);oldmetrics=metrics(prior);earliermetrics=metrics(older)
    oldquiet=float(np.mean([np.mean([r['physical_mae_model'] for r in rr if not r['motion_rich']]) for rr in specialist]))
    seedresults=[{'seed':s,'horizons':{str(h):summarize([r for r in rr if r['horizon']==h]) for h in HORIZONS}} for s,rr in zip(SEEDS,raw)]
    merged=averaged(raw);oldmerged=averaged(prior)
    tasks={task:summarize([r for r in merged if r['horizon']==1 and r['task']==task]) for task in sorted({r['task'] for r in merged})}
    for value in tasks.values():value.pop('event_ranking',None)
    episodes=sorted({(r['task'],r['episode']) for r in merged});draws=np.random.default_rng(80080).integers(len(episodes),size=(5000,len(episodes)))
    newtotals=np.array([[sum(r['mae'] for r in merged if r['horizon']==1 and (r['task'],r['episode'])==ep),sum(r['horizon']==1 and (r['task'],r['episode'])==ep for r in merged)] for ep in episodes])
    oldtotals=np.array([[sum(r['mae'] for r in oldmerged if r['horizon']==1 and (r['task'],r['episode'])==ep),sum(r['horizon']==1 and (r['task'],r['episode'])==ep for r in oldmerged)] for ep in episodes])
    ns=newtotals[draws].sum(1);os=oldtotals[draws].sum(1)
    ci=np.quantile(ns[:,0]/ns[:,1]-os[:,0]/os[:,1],[.025,.975]).tolist()
    wins=sum(np.mean([r['mae'] for r in n if r['horizon']==1])<np.mean([r['mae'] for r in o if r['horizon']==1]) for n,o in zip(raw,prior))
    gates={'H1at_least5pct_below_ME079':newmetrics['1']['all']['mae']<=.95*oldmetrics['1']['all']['mae'],
      'H1moving_within5pct_ME078':newmetrics['1']['moving']['mae']<=1.05*earliermetrics['1']['moving']['mae'],
      'H1stationary_within10pct_ME077':newmetrics['1']['stationary']['mae']<=1.10*oldquiet,
      'no_task_H1over10pct_persistence':all(r['mae']<=1.10*r['persistence_mae'] for r in tasks.values()),
      'paired_episode95CI_upper_below0':ci[1]<0,'at_least4of5paired_seed_H1wins':wins>=4}
    for h in HORIZONS:
        gates[f'H{h}_coverage_at_least85pct']=newmetrics[str(h)]['all']['coverage']>=.85
        gates[f'H{h}_interval_score_no_worse_ME079']=newmetrics[str(h)]['all']['interval_score']<=oldmetrics[str(h)]['all']['interval_score']
        if h!=1:gates[f'H{h}_MAEwithin5pct_ME079']=newmetrics[str(h)]['all']['mae']<=1.05*oldmetrics[str(h)]['all']['mae']
    if all(gates.values()):tier='DEV_GAIN'
    elif newmetrics['1']['all']['mae']<oldmetrics['1']['all']['mae']:tier='MIXED_TRADEOFF'
    else:tier='REGRESSION'
    result={'tier':tier,'full_registered_gate':'PASS' if all(gates.values()) else 'FAIL','gates':gates,
      'native_ME080':newmetrics,'frozen_ME079_native':oldmetrics,'frozen_ME078_native':earliermetrics,
      'frozen_ME077_H1_stationary_MAE':oldquiet,'per_seed':seedresults,'tasks_H1':tasks,
      'paired_episode95CI_H1new_minus_ME079':ci,'paired_seed_H1wins_vs_ME079':wins,
      'real_episode_clusters':len(episodes),'future_pairs':len(raw[0]),'original_model_forecast_records':sum(len(rr) for rr in raw),
      'denominator_note':'same exposed40episodes; seed repeats/horizons do not create independent cases; site/operator independence unestablished',
      'no_architecture_winner':True,'no_new_holdout':True,'customer_operator_value':'NOT_MEASURED'}
    path=ROOT/'CAPABILITY_RESULTS.json'
    if path.exists() and load(path)!=result:raise RuntimeError('Never overwrite original results')
    if not path.exists():write(path,result)
    print(json.dumps({'tier':tier,'gates':gates,'H1':newmetrics['1'],'H4':newmetrics['4']['all'],'H16':newmetrics['16']['all'],'H1_CI':ci},indent=2))

if __name__=='__main__':main()
