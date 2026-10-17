"""Frozen-metric DEV-054 reporting; suppress unstable normalization."""
import csv
import json
import math
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from tac_osm.development import joint_repair as j
from tac_osm.development.research_statistics import crossed_draws
from run_learned_belief_dev052b import write


def summary(directory, output):
    directory, out = Path(directory), Path(output); out.mkdir(parents=True, exist_ok=True)
    common = json.loads((directory/'common.json').read_text())
    assert common['status'] == 'COMPLETE_UNAUDITED' and not common['failures']
    reports = [json.loads(p.read_text()) for p in sorted(directory.glob('seed*/report.json'))]
    def cells(arm, family):
        return [r['arms'][arm]['evaluation'][family] for r in reports] if arm in j.NEURAL else [common['baselines'][arm][family]]
    def matrix(arm, family, field='cost'):
        divisor = 2 if field == 'cost' else 1
        return np.array([[r[field]/divisor for r in c['rows']] for c in cells(arm,family)]).reshape(len(cells(arm,family)), -1, 3)
    def interval(x, b, oracle=None, strata=None):
        w, ids = crossed_draws(x.shape[0], x.shape[1], strata, 5000, 5540002)
        draws = np.take_along_axis(w@x, ids, 1).mean(1)
        delta = draws-b[ids].mean(1)
        result = {'cost': float(x.mean()), 'delta_vs_beta': float((x-b).mean()), 'delta_95pct': np.quantile(delta,[.025,.975]).tolist()}
        if oracle is not None:
            gap = float((b-oracle).mean()); denominator = (b[ids]-oracle[ids]).mean(1)
            invalid = float((denominator<=0).mean())
            eligible = gap >= .05 and invalid < .01
            result.update({'oracle_headroom':gap,'invalid_denominator_fraction':invalid,'ratio_eligible':eligible,
                           'fraction_gap_closed': float((b-x.mean(0)).mean()/gap) if eligible else None,
                           'fraction_gap_95pct':np.quantile((b[ids].mean(1)-draws)/denominator,[.025,.975]).tolist() if eligible else None})
        return result
    result = {'study':'DEV-054','source':common['source'],'pooled':{},'by_family':{},'contrasts':{},'training':{},'calibration':{},
              'independent_streams':256,'model_seeds':[r['seed'] for r in reports], 'status':'DESCRIPTIVE_DEVELOPMENT',
              'scope':'Learned regime posterior plus explicit joint physics/planner, MTSK only; no full TAC-OSM/CDL/CASM claim'}
    curves=[]
    for arm in j.BASELINES+j.NEURAL:
        x = np.concatenate([matrix(arm,f).mean(-1) for f in ('switching','drift')],1)
        b = np.concatenate([matrix('stationary_beta',f)[0].mean(-1) for f in ('switching','drift')])
        o = np.concatenate([matrix('oracle',f)[0].mean(-1) for f in ('switching','drift')])
        strata = [0]*(len(b)//2)+[1]*(len(b)//2)
        result['pooled'][arm] = interval(x,b,o,strata)
        fx=np.concatenate([matrix(arm,f,'failures').mean(-1) for f in ('switching','drift')],1)
        fb=np.concatenate([matrix('stationary_beta',f,'failures')[0].mean(-1) for f in ('switching','drift')])
        fs=interval(fx,fb,strata=strata)
        result['pooled'][arm]['paired_faults_per_world']={'difference':fs['delta_vs_beta'],'interval_95pct':fs['delta_95pct']}
        result['calibration'][arm] = {f:[c['calibration'] for c in cells(arm,f)] for f in ('switching','drift')}
        for family in ('switching','drift'):
            result['by_family'].setdefault(family,{})[arm]={}
            for i,n in enumerate(j.LEVELS):
                x = matrix(arm,family)[:,:,i]; b=matrix('stationary_beta',family)[0,:,i]; o=matrix('oracle',family)[0,:,i]
                stat=interval(x,b,o); saving=-stat['delta_vs_beta']; acquisition=.6*n
                fs=interval(matrix(arm,family,'failures')[:,:,i],matrix('stationary_beta',family,'failures')[0,:,i])
                stat['paired_faults_per_world']={'difference':fs['delta_vs_beta'],'interval_95pct':fs['delta_95pct']}
                stat.update({'acquisition_per_entity':acquisition,'total_cost_per_entity':stat['cost']+acquisition,
                             'oracle_break_even_decisions':math.ceil(acquisition/stat['oracle_headroom']) if stat['oracle_headroom']>0 else None,
                             'model_break_even_decisions':math.ceil(acquisition/saving) if saving>0 else None,
                             'faults_mean_per_seed':float(matrix(arm,family,'failures')[:,:,i].sum(1).mean()),
                             'blocked':int(matrix(arm,family,'blocked')[:,:,i].sum()),'violations':int(matrix(arm,family,'violations')[:,:,i].sum())})
                result['by_family'][family][arm][str(n)]=stat
                curves.append({'family':family,'arm':arm,'observations':n, **{k:stat[k] for k in ('cost','delta_vs_beta','oracle_headroom','acquisition_per_entity','total_cost_per_entity','faults_mean_per_seed','ratio_eligible','fraction_gap_closed','oracle_break_even_decisions','model_break_even_decisions')}})
    for reference in ('gru_matched','gru128','gru_hmm','mtsk_reset'):
        diff=np.concatenate([(matrix('mtsk',f)-matrix(reference,f)).mean(-1) for f in ('switching','drift')],1)
        w,ids=crossed_draws(*diff.shape,strata,5000,5540003)
        draws=np.take_along_axis(w@diff,ids,1).mean(1)
        guards={}
        for f in ('switching','drift'):
            guards[f]=[bool((matrix('mtsk',f,'failures')[i].sum()<=matrix(reference,f,'failures')[i].sum()) and
                            matrix('mtsk',f,'blocked')[i].sum()<=matrix(reference,f,'blocked')[i].sum() and
                            matrix('mtsk',f,'violations')[i].sum()<=matrix(reference,f,'violations')[i].sum()) for i in range(len(reports))]
        adjusted=np.quantile(draws,[.05/8,1-.05/8]).tolist()
        result['contrasts'][reference]={'mtsk_minus_reference':float(diff.mean()),'bonferroni_four_95pct':adjusted,'reliability_by_seed_family':guards,'cost_reliability_guard':adjusted[1]<0 and all(all(v) for v in guards.values())}
    result['cost_reliability_supported']=all(result['contrasts'][a]['cost_reliability_guard'] for a in ('gru_matched','gru_hmm'))
    result['deployment_stream_measured']=False
    result['robust_mtsk_advantage']=result['cost_reliability_supported'] and result['deployment_stream_measured']
    result['commercial_guard_note']='Acquisition amortization is hypothetical; no persistent production decision stream was measured.'
    for r in reports:
        result['training'][str(r['seed'])]={a:c['training'] for a,c in r['arms'].items()}
    write(out/'summary.json',result)
    with (out/'curves.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(curves[0]));writer.writeheader();writer.writerows(curves)
    fig,axes=plt.subplots(1,2,figsize=(11,4),sharey=True)
    for ax,family in zip(axes,('switching','drift')):
        for arm in j.BASELINES+j.NEURAL[:-1]:
            ax.plot(j.LEVELS,[result['by_family'][family][arm][str(n)]['cost'] for n in j.LEVELS],marker='o',label=arm)
        ax.set(title=family,xlabel='Paid public observations',ylabel='Native cost / entity')
        ax.grid(alpha=.2)
    axes[1].legend(fontsize=8);fig.tight_layout();fig.savefig(out/'cost-curves.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for arm in j.NEURAL:
        cs=[r['arms'][arm]['training']['curve'] for r in reports]
        axes[0].plot([p['update'] for p in cs[0]],np.array([[p['validation_ce'] for p in c] for c in cs]).mean(0),label=arm)
        axes[1].plot([p['update'] for p in cs[0]],np.array([[p['train_ce'] for p in c] for c in cs]).mean(0),label=arm)
    axes[0].set(title='Validation (seed mean)',xlabel='Optimizer updates',ylabel='Soft regime cross entropy')
    axes[1].set(title='Training (seed mean)',xlabel='Optimizer updates');axes[1].legend(fontsize=8)
    fig.tight_layout();fig.savefig(out/'training-curves.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for ax,family in zip(axes,('switching','drift')):
        ax.plot([0,1],[0,1],color='gray',ls='--')
        for arm in ('mtsk','gru_matched','gru_hmm','hmm'):
            bins=result['calibration'][arm][family][0]['bins']
            ax.plot([b['prediction'] for b in bins if b['count']],[b['target_probability'] for b in bins if b['count']],marker='o',label=arm)
        ax.set(title=family+' (first seed)',xlabel='Predicted repair success',ylabel='Mean true success probability');ax.legend(fontsize=8)
    fig.tight_layout();fig.savefig(out/'calibration.png',dpi=180);plt.close(fig)
    print(json.dumps({'pooled':result['pooled'],'contrasts':result['contrasts']},indent=2))


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--directory',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();summary(a.directory,a.output)
