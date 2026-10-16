"""Reproduce a diagonal discrepancy without summarizing off-diagonal outcomes."""
import argparse
import json
from pathlib import Path
import platform
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
sys.path.insert(0,str(ROOT/'scripts'))
import torch  # noqa: E402
from tac_osm.development.entity_aging import EntityAgingControl  # noqa: E402
from tac_osm.development.noisy_memory import parameter_hash  # noqa: E402
from run_memory_controls_dev import evaluate  # noqa: E402
from audit_dev025_cross_eval import inputs, spec_and_sources, metrics, digest  # noqa: E402


def diagnose(parents, cohort, regime, seed, output):
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    arm='masked_local' if cohort=='DEV023' else 'joint'
    parent=next(parents.glob(f'*-{regime}-{seed}-{arm}-NOT-CONFIRMATORY'))
    original, bound=inputs(parent,f'{cohort}-{regime}-{seed}')
    model=EntityAgingControl('masked_local',seed)
    model.load_state_dict(torch.load(parent/'model.pt',weights_only=True,map_location='cpu'))
    assert parameter_hash(model)==bound['parameter_hash']
    spec=spec_and_sources()
    spec.update(h_levels=[512],eval_steps=4)
    rows,_=evaluate(model,seed+(23000 if cohort=='DEV023' else 24000),spec,False)
    archived=[r for r in original['rows'] if r['condition']['history']==512]
    floating=[]
    discrete=[]
    def scan(a,b,path=''):
        assert type(a) is type(b)
        if isinstance(a,dict):
            assert a.keys()==b.keys()
            for k in a:scan(a[k],b[k],path+'/'+str(k))
        elif isinstance(a,list):
            assert len(a)==len(b)
            for k,(x,y) in enumerate(zip(a,b)):scan(x,y,path+'/'+str(k))
        elif a!=b:
            (floating if isinstance(a,float) else discrete).append((abs(a-b),path,a,b) if isinstance(a,float) else (path,a,b))
    scan(rows,archived)
    fresh, old=metrics(rows),metrics(archived)
    result=dict(study='DEV-025-ENGINEERING-DIAGNOSTIC',cohort=cohort,regime=regime,seed=seed,torch=torch.__version__,python=platform.python_version(),input=bound,
                source_sha256={p:digest(ROOT/p) for p in ('scripts/diagnose_dev025_diagonal.py','scripts/run_memory_controls_dev.py','scripts/audit_dev025_cross_eval.py')},
                compared_rows=len(rows),float_differences=len(floating),max_absolute_float_difference=max(floating) if floating else None,
                discrete_differences=discrete,metric_differences={k:[fresh[k],old[k]] for k in fresh if fresh[k]!=old[k]})
    output.write_text(json.dumps(result,indent=2)+'\n')
    print('DIAGONAL_DIAGNOSTIC_RECORDED',output)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--parents',type=Path,required=True)
    p.add_argument('--cohort',choices=('DEV023','DEV024'),required=True)
    p.add_argument('--regime',choices=('short','long'),required=True)
    p.add_argument('--seed',type=int,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    diagnose(a.parents,a.cohort,a.regime,a.seed,a.output)
