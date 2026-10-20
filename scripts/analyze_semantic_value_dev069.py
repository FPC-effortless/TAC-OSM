"""Registered controller-only decision; no retuning or fresh source access."""
import argparse
import json
from pathlib import Path
import numpy as np
from run_learned_stack_dev055 import write
from analyze_public_outcome_dev068 import aggregate
from run_semantic_value_dev069 import eligible


def paired_cost(candidate,control):
    """Seeds AND paired world IDs resampled; same IDs across all histories."""
    delta=[]
    for left,right in zip(candidate,control,strict=True):
        assert left['seed']==right['seed']
        assert [c['history'] for c in left['primary']]==[c['history'] for c in right['primary']]
        delta.append(np.mean([[a['cost']-b['cost'] for a,b in zip(x['rows'],y['rows'],strict=True)]
            for x,y in zip(left['primary'],right['primary'],strict=True)],axis=0))
    delta=np.asarray(delta);assert delta.shape==(20,64)
    rng=np.random.default_rng(969003);means=[]
    for _ in range(10000):
        seeds=rng.integers(20,size=20);worlds=rng.integers(64,size=64)
        means.append(float(delta[seeds][:,worlds].mean()))
    upper=float(np.quantile(means,.95))
    return dict(mean_difference=float(delta.mean()),one_sided95_upper=upper,
        noninferiority_margin=.02,cost_noninferior=upper<=.02,
        bootstrap='10000 hierarchical paired seed/world resamples; shared world draws across histories',rng=969003)


def run(directory,output,interventions=None):
    root=Path(directory);summary=json.loads((root/'summary.json').read_text());per_fit={}
    for path in sorted(root.glob('*/*-report.json')):
        r=json.loads(path.read_text());item=dict(arm=r['arm'],seed=r['seed'],parent_seed=r['parent_seed'],
            readout=r['readout'],lr=r['lr'],parameters=r['parameters'],policy_parameters=r['policy_parameters'],
            validation=aggregate(r['validation']),validation_strata=[dict(history=c['history'],**aggregate([c])) for c in r['validation']],
            validation_entry=eligible(r))
        test=path.parent/f"{r['arm']}-test.json"
        if test.exists():
            t=json.loads(test.read_text());item.update(primary=aggregate(t['primary']),secondary=aggregate(t['secondary']),
                primary_strata=[dict(history=c['history'],**aggregate([c])) for c in t['primary']],
                secondary_strata=[dict(history=c['history'],**aggregate([c])) for c in t['secondary']])
        per_fit[str(path.relative_to(root))]=item
    result=dict(status='COMPLETE',run_status=summary['status'],fit_count=len(per_fit),per_fit=per_fit,
        tests_opened=summary['tests_opened'],source=summary['source'],whole_world_model_qualified=False,
        commercial_qualified=False,component_qualified=False)
    selected=root/'selection.json'
    if selected.exists():result['selection']=json.loads(selected.read_text())
    if summary['status'] in ('VALIDATION_REJECTED','ADMINISTRATIVELY_STOPPED'):
        assert not summary['tests_opened'] and not (root/'TEST-OPENED.json').exists()
        result.update(decision='NO_PROMOTION_INCOMPLETE_POLICY_STOP' if summary['status']=='ADMINISTRATIVELY_STOPPED' else 'NO_PROMOTION_VALIDATION_REJECTED',gates=dict(complete_registered_matrix=False,pilot_entry=selected.exists(),validation_reliability_qualified=False))
    elif summary['status']=='TEST_COMPLETE':
        primary=result['selection']['primary'];control=primary.removesuffix('_expected')+'_mlp'
        fits=[json.loads(p.read_text())|dict(report_path=str(p)) for p in sorted(root.glob('group-*/*-report.json'))]
        def rows(arm):
            values=[]
            for r in sorted((r for r in fits if r['arm']==arm),key=lambda r:r['seed']):
                p=Path(r['report_path']);t=json.loads((p.parent/f'{arm}-test.json').read_text())
                values.append(dict(seed=r['seed'],**t))
            assert len(values)==20;return values
        candidates=rows(primary);controls=rows(control)
        reliable=sum(all(c['completion']>=.95 for c in r['primary']+r['secondary']) for r in candidates)
        no_faults=all(c['faults']==0 for r in candidates for c in r['primary']+r['secondary'])
        no_extra=all(a['faults']<=b['faults'] for x,y in zip(candidates,controls,strict=True)
            for a,b in zip(x['primary']+x['secondary'],y['primary']+y['secondary'],strict=True))
        cost=paired_cost(candidates,controls)
        probe=json.loads(Path(interventions).read_text()) if interventions else {}
        semantic=probe.get('per_arm',{}).get(primary,{})
        gates=dict(reliable18=reliable>=18,zero_final_faults=no_faults,no_extra_faults=no_extra,
            cost_noninferior=cost['cost_noninferior'],known_world_probe_complete=probe.get('status')=='COMPLETE',
            known_world_degradation18=semantic.get('degradation_passing_seeds',0)>=18,
            known_world_no_extra_faults=semantic.get('no_extra_faults',False))
        result.update(primary=primary,control=control,individual_reliable_seeds=reliable,cost_contrast=cost,gates=gates,
            decision='READOUT_PERFORMANCE_GATES_PASS_PENDING_REPLAY' if all(gates.values()) else 'NO_PROMOTION_REGISTERED_GATES_FAILED')
    else:result['decision']='SMOKE_OR_INCOMPLETE_NOT_QUALIFIED'
    write(output,result);print(json.dumps({k:v for k,v in result.items() if k!='per_fit'}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--directory',required=True);p.add_argument('--output',required=True)
    p.add_argument('--interventions');a=p.parse_args();run(a.directory,a.output,a.interventions)
