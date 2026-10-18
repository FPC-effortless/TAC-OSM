"""Reproduce the separately registered exact-STE correction and physical guard."""
import argparse
import csv
import io
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
import torch
from run_retention_repair_dev061 import REPAIR,create,fixed_score
from run_structured_memory_dev061 import ROOT,raw_data,prepare,evaluate
from tac_osm.development.structured_memory import create as original_create
from run_learned_stack_dev055 import write,digest
from audit_memory_dev061 import compare,physical,standalone,PIN
from summarize_memory_dev061 import metrics,interval


def summarize(root,output):
    p=json.loads(REPAIR.read_text());s=json.loads((root/'summary.json').read_text());assert s['status']=='COMPLETE' and s['test_opened']
    evaluations=[json.loads((root/name).read_text()) for name in json.loads((root/'evaluations.json').read_text())]
    grouped={};rows=[]
    for r in evaluations:
        m=metrics(r['cells']);m['fixed_brier']=r['fixed']['brier'];m['pass']=m['completion']>.95;m['every_cell_pass']=all(np.mean([v['success'] for v in c['rows']])>.95 for c in r['cells'])
        grouped.setdefault(r['arm'],{}).setdefault(r['cohort'],{})[r['seed']]=m
        rows.append(dict(arm=r['arm'],cohort=r['cohort'],seed=r['seed'],**m))
    result={'status':'COMPLETE','protocol_sha256':digest(REPAIR),'run':s,'arms':{},'public_reference':metrics(json.loads((root/'public-reference.json').read_text())),
        'architectural_promotion':False,'scope':'Numerical correction on paired initialization seeds and fresh worlds. No full matched neural control grid on these new worlds; no architectural-superiority or corrected-OOD claim. Original flawed retention fits remain archived.'}
    for arm in p['arms']:
        before=grouped[arm]['original'];after=grouped[arm]['corrected'];entry={'per_seed':grouped[arm],
          'original_seeds_above95':sum(m['pass'] for m in before.values()),'corrected_seeds_above95':sum(m['pass'] for m in after.values()),
          'corrected_every_cell_pass':sum(m['every_cell_pass'] for m in after.values())}
        for key in ('completion','economic_cost','faults_per_world','fixed_brier','unnecessary_repairs'):
            entry[key]=interval([after[seed][key]-before[seed][key] for seed in p['seeds']])
        entry['physical_no_regression']=entry['faults_per_world']['paired_seed_bootstrap95'][1]<=0
        entry['cost_vs_public']=interval([after[seed]['economic_cost']-result['public_reference']['economic_cost'] for seed in p['seeds']])
        result['arms'][arm]=entry
    write(output,result)
    stream=io.StringIO();fields=['arm','cohort','seed','completion','pass','every_cell_pass','physical_repair','faults','economic_cost','actions','unnecessary_repairs','fixed_brier']
    writer=csv.DictWriter(stream,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(sorted(rows,key=lambda r:(r['seed'],r['arm'],r['cohort'])))
    Path(output).with_suffix('.csv').write_text(stream.getvalue())


@torch.no_grad()
def audit(original,directory,output,veritas=None,exports=None):
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);old=Path(original);root=Path(directory);p=json.loads(REPAIR.read_text())
    s=json.loads((root/'summary.json').read_text());assert s['status']=='COMPLETE' and s['test_opened'];assert s['protocol_sha256']==digest(REPAIR)
    for name,expected in json.loads((root/'manifest.json').read_text()).items():assert digest(root/name)==expected,name
    if veritas:
        assert subprocess.check_output(['git','-C',str(veritas),'rev-parse','HEAD'],text=True).strip()==PIN
        sys.path.insert(0,str(Path(veritas)/'src'))
    result={'status':'RUNNING','checkpoints':0,'cells':0,'trajectories':0,'actions':0,'native_actions':0,'failed_rollouts_all_arms_splits':0,'standalone_decisions':0,
            'exact_cached_incremental_actions':0,'exact_retained_timestamp_identities':0,'telemetry_drift':{'values':0,'max_absolute_difference':0.},
            'scope':'Shared synthetic physics, fresh correction test worlds; separate replay/portability, not external scientific validation.'}
    reports=json.loads((root/'fits.json').read_text())
    for source in {r['source'] for r in reports}:
        for name in ('src/tac_osm/development/exact_retention.py','scripts/run_retention_repair_dev061.py'):
            assert subprocess.check_output(['git','show',f'{source}:{name}'],cwd=ROOT)==(ROOT/name).read_bytes()
    fixed=raw_data(p['data']['fixed_query_seed'],p['data']['fixed_test_snapshots'])
    for r in reports:
        folder=root/f"{r['arm']}-{r['seed']}";checkpoint=folder/'agent.pt';assert digest(checkpoint)==r['checkpoint_sha256']
        model=create(r['arm'],r['config'],r['seed']).eval();model.load_state_dict(torch.load(checkpoint,weights_only=True),strict=True)
        prepared=prepare(model,fixed);cached=model.training_state(prepared);incremental=model.initial(len(fixed['events']))
        for t in range(fixed['events'].shape[1]):incremental,_=model.observe(incremental,fixed['events'][:,t],fixed['present'][:,t])
        torch.testing.assert_close(cached[0],incremental[0],atol=1e-5,rtol=1e-5);torch.testing.assert_close(cached[1],incremental[1],atol=0,rtol=0)
        a=model(cached,fixed['descriptors'],fixed['valid'],fixed['queries'],fixed['query_valid']);b=model(incremental,fixed['descriptors'],fixed['valid'],fixed['queries'],fixed['query_valid'])
        assert torch.equal(a['logits'].argmax(-1),b['logits'].argmax(-1));result['exact_cached_incremental_actions']+=len(fixed['events'])
        model.train();train_state=model.training_state(prepared);model.eval()
        for i in range(len(fixed['events'])):
            dates=train_state[0][i,16:32,-1][train_state[1][i,16:32]>0]
            for date in dates:
                assert ((prepared['signals'][i,:,-1]==date)&prepared['signal_present'][i]).any(),'counterfactual timestamp identity lost'
            result['exact_retained_timestamp_identities']+=len(dates)
        for cell in r['validation']:
            actual=evaluate(model,cell['seed'],cell['entities'],cell['history'],cell['worlds'],cell['family']);compare(actual,cell,result['telemetry_drift']);result['cells']+=1
            for row in cell['rows']:physical(row,result,bool(veritas))
        result['standalone_decisions']+=standalone(model,checkpoint,r['arm'],r['config'],Path(exports)/r['arm'] if exports and r['seed']==9101 else None)
        result['checkpoints']+=1
    for name in json.loads((root/'evaluations.json').read_text()):
        record=json.loads((root/name).read_text());arm,seed=record['arm'],record['seed'];cohort=record['cohort'];folder=(old if cohort=='original' else root)/f'{arm}-{seed}'
        factory=original_create if cohort=='original' else create
        model=factory(arm,record['config'],seed).eval();model.load_state_dict(torch.load(folder/'agent.pt',weights_only=True),strict=True)
        for cell in record['cells']:
            actual=evaluate(model,cell['seed'],cell['entities'],cell['history'],cell['worlds'],cell['family']);compare(actual,cell,result['telemetry_drift']);result['cells']+=1
            for row in cell['rows']:physical(row,result,bool(veritas))
        actual=fixed_score(model,fixed);np.testing.assert_allclose(actual['brier'],record['fixed']['brier'],atol=1e-4,rtol=1e-5);assert actual['error']==record['fixed']['error']
        print('correction audited',cohort,arm,seed,flush=True)
    result['status']='PASS';write(output,result)


if __name__=='__main__':
    a=argparse.ArgumentParser(description=__doc__);a.add_argument('--original',required=True);a.add_argument('--directory',required=True);a.add_argument('--output',required=True);a.add_argument('--veritas-root');a.add_argument('--export-root');a.add_argument('--summarize',action='store_true');args=a.parse_args()
    if args.summarize:summarize(Path(args.directory),args.output)
    else:audit(args.original,args.directory,args.output,args.veritas_root,args.export_root)
