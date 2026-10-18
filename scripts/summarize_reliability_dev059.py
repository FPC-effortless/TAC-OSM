"""Paired ten-seed and crossed seed/world summaries; no new model selection."""
import argparse
import itertools
import json
from pathlib import Path
import numpy as np
from run_learned_stack_dev055 import write


def signflip(values):
    d=np.asarray(values,float)
    if len(d)!=10:return None
    means=np.array([np.mean(d*np.asarray(sign)) for sign in itertools.product((-1,1),repeat=len(d))])
    return float(np.mean(np.abs(means)>=abs(d.mean())-1e-12))


def holm(values):
    order=sorted(values,key=values.get);result={};previous=0
    for rank,key in enumerate(order):
        previous=max(previous,min(1.,values[key]*(len(order)-rank)));result[key]=previous
    return result


def crossed_interval(difference,seed=8595901,repeats=2000):
    """Resample model seeds and paired worlds within each N/history cell."""
    a=np.asarray(difference);s,c,w=a.shape;rng=np.random.default_rng(seed);means=[]
    for _ in range(repeats):
        chosen=rng.integers(s,size=s);value=0.
        for cell in range(c):value+=a[chosen,cell][:,rng.integers(w,size=w)].mean()/c
        means.append(value)
    return np.quantile(means,[.025,.975]).tolist()


def matrix(directory,arm,seeds,metric):
    rows=[]
    for seed in seeds:
        report=json.loads((directory/f'{arm}-{seed}/test.json').read_text())
        cells=[]
        for c in report['cells']:
            values=[]
            for r in c['rows']:
                if metric=='completion':values.append(float(r['success']))
                elif metric=='repair':values.append(float(r['physical_repair']))
                elif metric=='actions':values.append(float(len(r['actions'])))
                elif metric=='completion_brier':values.append(float(np.mean((np.asarray(r['completion_probabilities'])-np.asarray(r['healthy_at_decision']))**2)))
                elif metric=='stop_brier':values.append(float(np.mean((np.asarray(r['stop_probabilities'])-np.asarray(r['healthy_at_decision']))**2)))
                else:values.append(float(r[metric]))
            cells.append(values)
        rows.append(cells)
    return np.asarray(rows)


def run(directory,output):
    root=Path(directory);summary=json.loads((root/'summary.json').read_text())
    reports=json.loads((root/'fits.json').read_text());arms=sorted({r['arm'] for r in reports})
    seeds=sorted({r['seed'] for r in reports});result={'status':summary['status'],'seeds':seeds,'arms':{},'comparisons':{},
        'unit':'Ten model initialization seeds and 576 shared stratified worlds. Not 28800 independent worlds.',
        'interval':'Crossed resampling of seed indices and world indices within9cells, paired across arms; 2000 replicates.',
        'limitations':'Fixed update/sample budget with different actual CPU/state and structural pretraining. No convergence, industrial reliability, global optimality or C5 claim.'}
    if summary['smoke'] or len(seeds)!=10 or summary['status']!='COMPLETE':
        result['comparison_status']='INCOMPLETE_NO_SUPERIORITY_CLAIM';write(output,result);return result
    metrics=('completion','repair','final_faults','cost','physical_action_cost','actions','healthy_nonstop_actions','harmful_regressions','writes','observations','completion_brier','stop_brier')
    arrays={arm:{metric:matrix(root,arm,seeds,metric) for metric in metrics} for arm in arms}
    for arm in arms:
        spec=json.loads((root/f'{arm}-{seeds[0]}/report.json').read_text())
        cells=json.loads((root/f'{arm}-{seeds[0]}/test.json').read_text())['cells']
        fitted=[r for r in reports if r['arm']==arm]
        result['arms'][arm]={'parameters':spec['parameters'],
            'persistent_state_bytes_per_world':cells[0]['persistent_state_bytes_per_world'],
            'total_training_cpu_seconds':sum(r['total_cpu_seconds'] for r in fitted),
            'per_seed':{metric:arrays[arm][metric].mean((1,2)).tolist() for metric in metrics},
            'means':{metric:float(arrays[arm][metric].mean()) for metric in metrics},
            'seed_sd_completion':float(arrays[arm]['completion'].mean((1,2)).std(ddof=1)),
            'per_cell_completion_mean_across_seeds':arrays[arm]['completion'].mean((0,2)).tolist()}
    pvalues={}
    for arm in arms:
        if arm=='full_stop':continue
        diffs={key:arrays['full_stop'][key]-arrays[arm][key] for key in metrics}
        result['comparisons'][arm]={'definition':'full_stop minus '+arm,
            'differences':{key:{'mean':float(d.mean()),'crossed95':crossed_interval(d),
                                'paired_seed95_t':[(float(d.mean())-2.262157*d.mean((1,2)).std(ddof=1)/np.sqrt(10)),
                                                   (float(d.mean())+2.262157*d.mean((1,2)).std(ddof=1)/np.sqrt(10))]}
                           for key,d in diffs.items()}}
        pvalues[arm]=signflip(diffs['completion'].mean((1,2)))
    adjusted=holm(pvalues)
    for arm in pvalues:result['comparisons'][arm].update(paired_seed_signflip_two_sided=pvalues[arm],holm_adjusted_p=adjusted[arm])
    controls=('no_casm_stop','gru_stop','attention_stop')
    result['superiority_gate']=all(result['comparisons'][a]['differences']['completion']['crossed95'][0]>0 and adjusted[a]<.05 and
                                   result['comparisons'][a]['differences']['final_faults']['crossed95'][1]<=0 for a in controls)
    result['stopping_change_supported']=result['comparisons']['full_common']['differences']['completion']['crossed95'][0]>0 and adjusted['full_common']<.05 and result['comparisons']['full_common']['differences']['final_faults']['crossed95'][1]<=0
    write(output,result)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--directory',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();run(args.directory,args.output)
