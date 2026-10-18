"""Registered numerical correction, no outcome-driven hyperparameter changes."""
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import json
import multiprocessing
from pathlib import Path
import time
import torch
import run_structured_memory_dev061 as base
from run_learned_stack_dev055 import write,digest,git
from tac_osm.development.exact_retention import ExactRetentionAgent
from tac_osm.development.structured_memory import create as original_create
from diagnose_memory_dev061 import reference
REPAIR=base.ROOT/'research/dev061/retention-repair/protocol.json'


def create(arm,config,seed=0):
    return ExactRetentionAgent(config,arm,seed) if 'retention' in arm else original_create(arm,config,seed)


def fit(job):
    base.create=create
    report=base.fit(job);report['implementation']='exact_retention';report['repair_protocol_sha256']=digest(REPAIR)
    write(Path(job[4])/'report.json',report);return report


@torch.no_grad()
def fixed_score(model,ds):
    model.eval();state=model.initial(len(ds['events']))
    for t in range(ds['events'].shape[1]):state,_=model.observe(state,ds['events'][:,t],ds['present'][:,t])
    out=model(state,ds['descriptors'],ds['valid'],ds['queries'],ds['query_valid']);q=ds['query_valid']
    errors=(out['current_logits'].sigmoid()-ds['current'])**2
    return {'brier':float(errors[q].mean()),'error':float(((out['current_logits']>=0)!=ds['current'].bool())[q].float().mean()),
            'per_world_brier':[float(errors[i,q[i]].mean()) for i in range(len(q))]}


def run(original,directory):
    torch.set_num_threads(1);p=json.loads(REPAIR.read_text());old=Path(original);root=Path(directory);root.mkdir(parents=True,exist_ok=False)
    s=json.loads((old/'summary.json').read_text());assert s['status']=='COMPLETE' and s['test_opened']
    assert not git('status','--porcelain','--untracked-files=no'),'Commit correction before fitting'
    specs=json.loads((old/'configurations.json').read_text());main_reports=json.loads((old/'fits.json').read_text())
    original_cpu=sum(r['cpu_seconds'] for r in main_reports+json.loads((old/'tuning.json').read_text()))
    jobs=[(arm,seed,specs[arm],p['lrs'][arm],str(root/f'{arm}-{seed}'),str(old/'_datasets'),False) for arm in p['arms'] for seed in p['seeds']]
    reports=[];start=time.monotonic()
    with ProcessPoolExecutor(max_workers=2,mp_context=multiprocessing.get_context('spawn')) as pool:
        for i in range(0,len(jobs),2):
            if original_cpu+sum(r['cpu_seconds'] for r in reports)>=10800:break
            for future in as_completed([pool.submit(fit,j) for j in jobs[i:i+2]]):
                r=future.result();reports.append(r);write(root/'fits.json',reports)
                print('corrected fit',r['arm'],r['seed'],r['status'],'cpu',round(r['cpu_seconds'],1),flush=True)
    complete=len(reports)==20 and all(r['status']=='COMPLETE' for r in reports) and original_cpu+sum(r['cpu_seconds'] for r in reports)<=10800
    summary={'status':'COMPLETE' if complete else 'INCOMPLETE','source':git('rev-parse','HEAD'),'protocol_sha256':digest(REPAIR),'fits':len(reports),'original_fitting_cpu_seconds':original_cpu,'correction_fitting_cpu_seconds':sum(r['cpu_seconds'] for r in reports),'test_opened':False}
    write(root/'summary.json',summary)
    if complete:
        fixed=base.raw_data(p['data']['fixed_query_seed'],p['data']['fixed_test_snapshots']);evaluations=[]
        reference_cells=[reference(p['data']['test_seed'],n,h,32,'clean') for n in (2,4,6) for h in (32,128,512,1024)]
        write(root/'public-reference.json',reference_cells)
        for cohort,records in [('original',[r for r in main_reports if r['arm'] in p['arms']]),('corrected',reports)]:
            for r in records:
                folder=(old if cohort=='original' else root)/f"{r['arm']}-{r['seed']}"
                model=(original_create if cohort=='original' else create)(r['arm'],r['config'],r['seed']).eval()
                model.load_state_dict(torch.load(folder/'agent.pt',weights_only=True),strict=True)
                cells=[base.evaluate(model,p['data']['test_seed'],n,h,32) for n in (2,4,6) for h in (32,128,512,1024)]
                output={'cohort':cohort,'arm':r['arm'],'seed':r['seed'],'config':r['config'],'cells':cells,'fixed':fixed_score(model,fixed)}
                name=f"evaluation-{cohort}-{r['arm']}-{r['seed']}.json";write(root/name,output);evaluations.append(name)
                print('evaluated',cohort,r['arm'],r['seed'],flush=True)
        write(root/'evaluations.json',evaluations);summary['test_opened']=True;summary['elapsed_seconds']=time.monotonic()-start;write(root/'summary.json',summary)
    write(root/'manifest.json',{str(f.relative_to(root)):digest(f) for f in sorted(root.rglob('*')) if f.is_file() and f.name!='manifest.json'})


if __name__=='__main__':
    a=argparse.ArgumentParser(description=__doc__);a.add_argument('--original',required=True);a.add_argument('--directory',required=True);args=a.parse_args();run(args.original,args.directory)
