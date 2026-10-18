"""Collect cross-CPU discrepancies without relaxing the strict replay audit.

Diagnostic completion is not a replay PASS. The preceding strict CI step remains
failed. Preserve actual public rollouts, discrete decision changes and physical
outcomes separately from floating telemetry failures; never refit or alter a model.
"""
import argparse
import json
from pathlib import Path
import platform
import sys
import torch
from audit_memory_dev061 import compare,physical,PIN
from evaluate_memory_fast_dev061 import evaluate
from run_structured_memory_dev061 import create
from run_learned_stack_dev055 import write,digest
from summarize_memory_dev061 import metrics


@torch.no_grad()
def run(original,corrected,output,veritas=None,limit=None):
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    if veritas:
        import subprocess
        assert subprocess.check_output(['git','-C',veritas,'rev-parse','HEAD'],text=True).strip()==PIN
        sys.path.insert(0,str(Path(veritas)/'src'))
    old,new=Path(original),Path(corrected);tasks=[]
    for root in (old,new):
        for name,expected in json.loads((root/'manifest.json').read_text()).items():assert digest(root/name)==expected
        folders={}
        for path in root.glob('*/report.json'):
            r=json.loads(path.read_text());folders[r['arm'],r['seed'],r['lr']]=path.parent
        reports=json.loads((root/'fits.json').read_text())
        if root==old:reports=json.loads((root/'tuning.json').read_text())+reports
        for r in reports:
            folder=folders[r['arm'],r['seed'],r['lr']]
            assert digest(folder/'agent.pt')==r['checkpoint_sha256']
            tasks.append((folder,r,r['validation'],'validation',False))
            if (folder/'test.json').exists():tasks.append((folder,r,json.loads((folder/'test.json').read_text())['cells'],'original-test',True))
    for name in json.loads((new/'evaluations.json').read_text()):
        r=json.loads((new/name).read_text());folder=(old if r['cohort']=='original' else new)/f"{r['arm']}-{r['seed']}"
        tasks.append((folder,r,r['cells'],'correction-'+r['cohort'],True))
    result={'status':'RUNNING','scope':'Diagnostic only. Strict audit tolerances and action requirements unchanged; this cannot turn failed replay CI green.',
       'hardware':platform.platform(),'cpu_model':next((v.split(':',1)[1].strip() for v in Path('/proc/cpuinfo').read_text().splitlines() if v.startswith('model name')),'UNKNOWN'),
       'maximum_aligned_float_difference':0.,'telemetry_length_mismatches':0,'tasks':0,'cells':0,'strict_failure_cells':0,'changed_action_worlds':0,'changed_completion_worlds':0,'changed_final_fault_worlds':0,
       'trajectories':0,'actions':0,'native_actions':0,'failed_rollouts_all_arms_splits':0,'telemetry_drift':{'values':0,'max_absolute_difference':0.},'primary_per_seed':[],'failures':[]}
    raw=Path(output).with_suffix('');raw.mkdir(parents=True,exist_ok=True)
    for number,(folder,r,cells,cohort,primary) in enumerate(tasks[:limit] if limit else tasks):
        model=create(r['arm'],r['config'],r['seed']).eval();model.load_state_dict(torch.load(folder/'agent.pt',weights_only=True),strict=True)
        actual_cells=[]
        for c in cells:
            actual=evaluate(model,c['seed'],c['entities'],c['history'],c['worlds'],c['family']);actual_cells.append(actual);result['cells']+=1
            try:compare(actual,c,result['telemetry_drift'])
            except (AssertionError,ValueError) as error:
                result['strict_failure_cells']+=1;result['failures'].append({'arm':r['arm'],'seed':r['seed'],'cohort':cohort,'cell':{k:c[k] for k in ('seed','entities','history','family')},'reason':str(error)[:1200]})
            for a,b in zip(actual['rows'],c['rows'],strict=True):
                for key in ('completion_probability','stop_probability','current_brier'):
                    if len(a[key])!=len(b[key]):result['telemetry_length_mismatches']+=1
                    else:result['maximum_aligned_float_difference']=max(result['maximum_aligned_float_difference'],max((abs(x-y) for x,y in zip(a[key],b[key],strict=True)),default=0.))
                result['changed_action_worlds']+=int([(v['kind'],v['target']) for v in a['actions']]!=[(v['kind'],v['target']) for v in b['actions']])
                result['changed_completion_worlds']+=int(a['success']!=b['success']);result['changed_final_fault_worlds']+=int(a['final_faults']!=b['final_faults'])
                physical(a,result,bool(veritas))
        write(raw/f'{number:03d}-{cohort}-{r["arm"]}-{r["seed"]}.json',{'checkpoint_sha256':digest(folder/'agent.pt'),'cohort':cohort,'cells':actual_cells})
        if primary:
            clean=[c for c in actual_cells if c['family']=='clean'];m=metrics(clean)
            result['primary_per_seed'].append({'arm':r['arm'],'seed':r['seed'],'cohort':cohort,**{k:m[k] for k in ('completion','physical_repair','faults','economic_cost')}})
        result['tasks']+=1
        print('CPU diagnostic',cohort,r['arm'],r['seed'],'strict failure cells',result['strict_failure_cells'],'changed action worlds',result['changed_action_worlds'],flush=True)
    result['status']='SMOKE_ONLY' if limit else 'QUALIFIED_FAILURE' if result['strict_failure_cells'] else 'PASS'
    write(output,result)
    print('CPU_DIAGNOSTIC_SUMMARY '+json.dumps({k:v for k,v in result.items() if k not in ('failures','primary_per_seed')},sort_keys=True),flush=True)
    print('CPU_DIAGNOSTIC_PRIMARY '+json.dumps(result['primary_per_seed'],sort_keys=True),flush=True)


if __name__=='__main__':
    a=argparse.ArgumentParser(description=__doc__);a.add_argument('--original',required=True);a.add_argument('--corrected',required=True);a.add_argument('--output',required=True);a.add_argument('--veritas-root');a.add_argument('--limit',type=int);v=a.parse_args();run(v.original,v.corrected,v.output,v.veritas_root,v.limit)
