"""No-fit paired ME102/ME103 summary and original source-label audit."""
from pathlib import Path
import sys,json,hashlib
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from scripts import me100_sensor_data as data
from scripts.me102_panel_summary import _metric_row,_check_metrics,_mean_dict,_numeric_replay
from scripts.me103_continuation_panel import OUT,RECIPES,SEEDS
DEST=OUT/'PANEL_SUMMARY.json'

def summary():
    child=json.loads((OUT/'RESULTS.json').read_text())
    oldroot=ROOT/'research/replay-bundles/ME102';parent=json.loads((oldroot/'RESULTS.json').read_text())
    new=np.load(OUT/'FORECASTS.npz',allow_pickle=False);old=np.load(oldroot/'FORECASTS.npz',allow_pickle=False)
    raw=data.load('report');cohorts={}
    for cohort,n,weeks_expected in (('gap',143,21),('general',4288,53)):
        for suffix in ('origins','truth','observed','locf','ridge'):
            np.testing.assert_array_equal(new[cohort+'_'+suffix],old[cohort+'_'+suffix])
        origins=new[cohort+'_origins'];truth=new[cohort+'_truth'];observed=new[cohort+'_observed'].astype(bool)
        np.testing.assert_array_equal(origins,np.asarray(raw[cohort]))
        rawtruth=raw['raw'][origins[:,0],origins[:,1]+6,:6]
        np.testing.assert_array_equal(observed,np.isfinite(rawtruth));np.testing.assert_array_equal(truth,np.nan_to_num(rawtruth,nan=0))
        weeks=origins[:,1]//168;clusters=np.unique(weeks)
        if len(origins)!=n or len(clusters)!=weeks_expected:raise RuntimeError('originalcohortcounts changed')
        recipes={}
        for recipe in RECIPES:
            nextrows=[];priorrows=[];differences=[]
            for seed in SEEDS:
                key=f'{cohort}_{recipe}_{seed}';q=new[key]
                if not np.isfinite(q).all() or (q<0).any() or (np.diff(q,axis=-1)<0).any():raise RuntimeError('invalid original quantiles')
                c=next(r for r in child['cohorts'][cohort]['models'] if r['recipe']==recipe and r['seed']==seed)
                p=next(r for r in parent['cohorts'][cohort]['models'] if r['recipe']==recipe and r['seed']==seed)
                cm=_metric_row(q,truth,observed,c['raw_density_nll']);pm=_metric_row(old[key],truth,observed,p['raw_density_nll'])
                _check_metrics(cm,c,f'ME103 {cohort} {recipe} {seed}');_check_metrics(pm,p,f'ME102 {cohort} {recipe} {seed}')
                nextrows.append(dict(seed=seed,**cm));priorrows.append(dict(seed=seed,**pm))
                differences.append(np.abs(q[:,0,1]-truth[:,0])-np.abs(old[key][:,0,1]-truth[:,0]))
            delta=np.asarray(differences).mean(axis=0);keep=observed[:,0]
            # Shared calendar weeks across stations; seeds condition on fitted weights.
            values={w:delta[(weeks==w)&keep] for w in clusters}
            def draw(selected):
                selected_values=[values[w] for w in selected]
                return float(np.concatenate(selected_values).mean())
            rng=np.random.default_rng(103103);ci=np.quantile([draw(rng.choice(clusters,len(clusters),replace=True)) for _ in range(2000)],[.025,.975])
            rows=[r for r in child['models'] if r['recipe']==recipe]
            oldrows=[r for r in parent['models'] if r['recipe']==recipe]
            recipes[recipe]=dict(continued_mean=_mean_dict(nextrows),parent_mean=_mean_dict(priorrows),continued_per_seed=nextrows,parent_per_seed=priorrows,full_gate_seed_count=sum(r['full_gate'] for r in rows),parent_full_gate_seed_count=sum(r['full_gate'] for r in oldrows),all_five_gate=child['recipe_all_seed_gates'][recipe],pm25_mae_difference_mean=draw(clusters),pm25_mae_difference_ci95=ci.tolist(),bootstrap_replicates=2000,bootstrap_scope='posthoc conditional on fitted weights; calendar-week clusters, not multiplicity-adjusted or independent confirmation')
        cohorts[cohort]=dict(queries=n,calendar_week_resampling_clusters=len(clusters),truth_sha256=hashlib.sha256(truth.tobytes()).hexdigest(),recipes=recipes)
    result=dict(status='EXPOSED_DEVELOPMENT_FIXED_CONTINUATION',no_new_fits=True,protected_2016_2017_scored=False,source_sha256=data.SHA,raw_source_labels_recomputed=True,cohorts=cohorts)
    if DEST.exists():_numeric_replay(result,json.loads(DEST.read_text()));print('ME103_PANEL_SUMMARY_REPLAY=PASS')
    else:
        with DEST.open('x') as f:json.dump(result,f,indent=2,allow_nan=False)
        print('ME103_PANEL_SUMMARY_SAVED=PASS')
    for recipe,row in cohorts['gap']['recipes'].items():
        c=row['continued_mean'];p=row['parent_mean'];print(json.dumps(dict(recipe=recipe,mae=c['by_channel']['0']['mae'],parent_mae=p['by_channel']['0']['mae'],coverage=c['by_channel']['0']['coverage'],nll=c['nll'],interval_score=c['by_channel']['0']['interval_score'],full_gate_seeds=row['full_gate_seed_count'],all5=row['all_five_gate'])))
if __name__=='__main__':summary()
