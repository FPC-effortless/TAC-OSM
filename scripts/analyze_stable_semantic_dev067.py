"""DEV067 validation screen or full fresh-seed evidence; never pick a test winner."""
import argparse
import csv
import io
import json
from pathlib import Path
import numpy as np
from analyze_belief_dev063 import metrics
from analyze_belief_dev062_r1 import reference
from summarize_memory_dev061 import interval, signflip, holm
from run_learned_stack_dev055 import write


def save_csv(output, rows):
    stream=io.StringIO();writer=csv.DictWriter(stream,fieldnames=list(rows[0]),lineterminator='\n')
    writer.writeheader();writer.writerows(rows);Path(output).with_suffix('.csv').write_text(stream.getvalue())


def run(directory, output):
    root=Path(directory);summary=json.loads((root/'summary.json').read_text())
    assert summary['status'] in ('VALIDATION_REJECTED','COMPLETE')
    protocol=json.loads(Path('research/dev067-stable-interface/protocol.json').read_text())
    specs=json.loads((root/'specs.json').read_text());selection=json.loads((root/'selection.json').read_text())
    pilots=[]
    for path in sorted(root.glob('pilot-*/*/report.json')):
        r=json.loads(path.read_text());m=metrics(r['validation']);f=r['validation_fixed']
        pilots.append(dict(arm=r['arm'],pilot=path.parent.parent.name,initialization=r['seed'],lr=r['lr'],
            closure=m['completion'],faults=m['faults'],cost=m['cost'],p95_KL=f['p95_KL'],p99_KL=f['p99_KL'],
            mean_KL=f['posterior_KL'],encoder_sha256=r['encoder_sha256']))
    parity=[]
    for folder in sorted([*root.glob('pilot-*'),*root.glob('group-*')]):
        reports=[json.loads(p.read_text()) for p in folder.glob('tac_*/report.json')]
        assert len(reports)==4 and len({r['encoder_sha256'] for r in reports})==1
        parity.append(dict(group=folder.name,encoder_sha256=reports[0]['encoder_sha256']))
    common=dict(run=summary,selection=selection,pilots=pilots,shared_TAC_encoder_groups=parity,
        scope='Six heads share three encoders per initialization. Probability/log and moving/frozen TAC heads have identical final encoders. Public-posterior supervision; no commercial qualification.')
    if summary['status']=='VALIDATION_REJECTED':
        assert not summary['test_opened'] and not (root/'TEST-EXPOSURE.json').exists()
        assert not list(root.glob('group-*/*/test.json')) and not (root/'test-fixed.pt').exists()
        write(output,dict(common,status='VALIDATION_REJECTED',decision='NO_PROMOTION_NO_FRESH20_SEED_CLAIM',
            test_evaluation='NOT_RUN_TEST_NAMESPACES_UNOPENED',commercial='NOT_ESTABLISHED'))
        save_csv(output,pilots);return
    assert summary['test_opened'] and (root/'TEST-EXPOSURE.json').exists()
    refs={kind:[reference(protocol['primary_seed'],h,64,kind) for h in protocol['histories']]
          for kind in ('public','privileged','shortcut')}
    secondary=[reference(protocol['secondary_seed'],32,64,'public',error=e,inspection=price)
               for e,price in ((.2,.1),(.1,.4))]
    write(Path(output).with_name('public-reference.json'),dict(refs,secondary_public=secondary))
    tables={};csvrows=[]
    for arm in specs:
        rows={}
        for seed in protocol['seeds']:
            test=json.loads((root/f'group-{seed}'/arm/'test.json').read_text());m=metrics(test['cells'][:4])
            m['strata']=[dict(history=h,**metrics([c])) for h,c in zip(protocol['histories'],test['cells'][:4],strict=True)]
            m['secondary']=[metrics([c]) for c in test['cells'][4:]];m['fixed']=test['fixed']
            m['public_cost_regret']=m['cost']-metrics(refs['public'])['cost']
            m['privileged_cost_regret']=m['cost']-metrics(refs['privileged'])['cost']
            actions=[a for c in test['cells'][:4] for row in c['rows'] for a in row['actions']]
            m['transient_fault_exposure']=sum(sum(a['after']) for a in actions)
            m['new_fault_bits_during_actions']=sum(sum(not b and v for b,v in zip(a['before'],a['after'],strict=True)) for a in actions)
            m['passes_each_history']=all(c['completion']>=.95 for c in m['strata'])
            m['passes_each_secondary']=all(c['completion']>=.95 for c in m['secondary'])
            m['individual_tail_pass']=m['fixed']['p95_KL']<=.05 and m['fixed']['p99_KL']<=.1
            m['physical_guard']=all(a['faults']<=b['faults'] for cell,ref in zip(test['cells'],refs['public']+secondary,strict=True) for a,b in zip(cell['rows'],ref['rows'],strict=True))
            rows[seed]=m
            csvrows.append(dict(arm=arm,seed=seed,**{k:m[k] for k in ('completion','physical_repair','faults','cost','actions','inspections','missing_stop','early_stops','stop_precision','stop_recall_at_healthy_decisions','late_healthy_actions','unnecessary_repairs','transient_fault_exposure','new_fault_bits_during_actions','public_cost_regret','privileged_cost_regret','passes_each_history','passes_each_secondary','individual_tail_pass','physical_guard')},
                p95_KL=m['fixed']['p95_KL'],p99_KL=m['fixed']['p99_KL'],secondary_faults=sum(c['faults'] for c in m['secondary'])))
        tables[arm]=rows
    aggregate={arm:dict(
        passing_seeds_each_history=sum(v['passes_each_history'] for v in rows.values()),
        passing_seeds_by_history={str(h):sum(v['strata'][i]['completion']>=.95 for v in rows.values()) for i,h in enumerate(protocol['histories'])},
        passing_seeds_each_secondary=sum(v['passes_each_secondary'] for v in rows.values()),
        physical_guard_seeds=sum(v['physical_guard'] for v in rows.values()),
        individual_tail_passes=sum(v['individual_tail_pass'] for v in rows.values()),
        faults=sum(v['faults'] for v in rows.values()),secondary_faults=sum(c['faults'] for v in rows.values() for c in v['secondary']),
        mean_completion=float(np.mean([v['completion'] for v in rows.values()])),
        mean_cost=float(np.mean([v['cost'] for v in rows.values()])),
        median_p95_KL=float(np.median([v['fixed']['p95_KL'] for v in rows.values()])),
        median_p99_KL=float(np.median([v['fixed']['p99_KL'] for v in rows.values()])),per_seed=rows)
        for arm,rows in tables.items()}
    def assessment(arm):
        values=aggregate[arm];comparisons={}
        for control in ('gru_frozen_log','no_casm_frozen_log'):
            delta=[tables[arm][s]['cost']-tables[control][s]['cost'] for s in protocol['seeds']]
            comparisons[control]=dict(cost=interval(delta),p=signflip(delta))
        for name,value in holm({n:v['p'] for n,v in comparisons.items()}).items():comparisons[name]['holm_p']=value
        return dict(arm=arm,comparisons=comparisons,gates=dict(
            reliability18_each_history=values['passing_seeds_each_history']>=18,
            reliability18_each_secondary=values['passing_seeds_each_secondary']>=18,
            physical_no_excess_all20=values['physical_guard_seeds']==20,
            individual_tails18of20=values['individual_tail_passes']>=18,
            cost_noninferiority002=all(v['cost']['paired_seed_bootstrap95'][1]<=.02 for v in comparisons.values()),
            replay_and_interventions='PENDING_SEPARATE_AUDIT'))
    effects={}
    coefficients=dict(
        frozen={'tac_frozen_probability':.5,'tac_frozen_log':.5,'tac_moving_probability':-.5,'tac_moving_log':-.5},
        logarithmic={'tac_moving_log':.5,'tac_frozen_log':.5,'tac_moving_probability':-.5,'tac_frozen_probability':-.5},
        interaction={'tac_frozen_log':1,'tac_moving_probability':1,'tac_frozen_probability':-1,'tac_moving_log':-1})
    for name,coefs in coefficients.items():
        for metric in ('completion','cost'):
            delta=[sum(c*tables[a][s][metric] for a,c in coefs.items()) for s in protocol['seeds']]
            effects[name+'/'+metric]=dict(estimate=interval(delta),p=signflip(delta))
    for name,value in holm({n:v['p'] for n,v in effects.items()}).items():effects[name]['holm_p']=value
    delta=[tables[protocol['primary']][s]['completion']-tables['tac_moving_probability'][s]['completion'] for s in protocol['seeds']]
    mechanism=interval(delta);mechanism['practical_5pp_gain']=mechanism['difference']>=.05 and mechanism['paired_seed_bootstrap95'][0]>0
    result=dict(common,status='COMPLETE',primary=assessment(protocol['primary']),
        preselected_candidate=assessment(selection['candidate']),arms=aggregate,factorial_effects=effects,
        mechanism_completion_gain=mechanism,references={k:metrics(v) for k,v in refs.items()},
        secondary_public=[metrics([c]) for c in secondary],commercial='NOT_ESTABLISHED')
    write(output,result);save_csv(output,csvrows)
    print(json.dumps(dict(primary=result['primary']['gates'],candidate=result['preselected_candidate']['gates'])),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--directory',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();run(a.directory,a.output)
