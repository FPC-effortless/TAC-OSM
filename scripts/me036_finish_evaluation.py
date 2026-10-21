"""Resume frozen scoring by model, preserving every completed metric exactly."""
import concurrent.futures,json,multiprocessing
import torch
import scripts.me036_research as b
from scripts.me036b_continue import OUT,Ridge
from tac_osm.me036_patch_forecast import build_model
import numpy as np

CACHE={}


class Persistence(torch.nn.Module):
    def __init__(self,h):super().__init__();self.h=h
    def forward(self,x):return x[:,-1:].expand(-1,self.h,-1)


def task_key(kind,row):
    return row['key'] if kind=='models' else f"{kind}-{row['dataset']}-h{row['horizon']}"


def score_task(task):
    torch.set_num_threads(1)
    kind,r,freeze=task;key=task_key(kind,r)
    name=r['dataset'];h=r['horizon']
    if name not in CACHE:CACHE[name]=b.load_data(name)
    z,meta=CACHE[name];_,val,end=meta['split_edges']
    if kind=='models':
        model=build_model(r['arm'],h,r['seed'])
        model.load_state_dict(torch.load(b.ROOT/r['selected']['checkpoint'],weights_only=False)['state_dict'])
        row={k:r[k] for k in ('key','dataset','horizon','arm','seed','parameters','charged_training_CPU_seconds')}
        row['selected_step']=r['selected']['step']
    elif kind=='ridge':model=Ridge(np.load(b.ROOT/r['coefficient']));row=dict(r)
    else:model=Persistence(h);row=dict(r)
    metric=b.score(model,z,val,end,h);row['test']=metric
    receipt={'kind':kind,'row':row,'freeze_sha256':freeze}
    b.dump(OUT/'scores'/(key+'.json'),receipt)
    print('SCORED',key,round(metric['MSE'],6),round(metric['CPU_seconds'],2),flush=True)
    return receipt


def run():
    torch.set_num_threads(1);frozen=json.loads((OUT/'frozen.json').read_text());freeze=b.sha(OUT/'frozen.json')
    code=json.loads((b.REPORT/'code-freeze.json').read_text())
    for item in code['files']:assert b.sha(b.ROOT/item['path'])==item['sha256']
    for r in frozen['fits']:
        for e in r['endpoints']:assert b.sha(b.ROOT/e['checkpoint'])==e['sha256']
    for r in frozen['ridge']:assert b.sha(b.ROOT/r['coefficient'])==r['sha256']
    completed={}
    # All prior completed values are immutable, including partial-cohort progress.
    for name in b.DATASETS:
        final=OUT/f'test-{name}.json';progress=OUT/f'test-progress-{name}.json'
        path=final if final.exists() else progress
        if not path.exists():continue
        old=json.loads(path.read_text());assert old['freeze_sha256']==freeze
        for kind in ('models','ridge','persistence'):
            for row in old.get(kind,[]):completed[task_key(kind,row)]={'kind':kind,'row':row,'freeze_sha256':freeze}
    score_dir=OUT/'scores';score_dir.mkdir(exist_ok=True)
    for rec in completed.values():b.dump(score_dir/(task_key(rec['kind'],rec['row'])+'.json'),rec)
    for path in score_dir.glob('*.json'):
        rec=json.loads(path.read_text());assert rec['freeze_sha256']==freeze
        completed[task_key(rec['kind'],rec['row'])]=rec
    tasks=[('models',r,freeze) for r in frozen['fits'] if r['status']=='PASS']
    tasks += [('ridge',r,freeze) for r in frozen['ridge']]
    tasks += [('persistence',{'dataset':d,'horizon':h},freeze) for d in b.DATASETS for h in b.HORIZONS]
    remaining=[t for t in tasks if task_key(t[0],t[1]) not in completed]
    # Long jobs first, two single-thread workers; every task uses original scoring function.
    remaining.sort(key=lambda t:t[1]['horizon'],reverse=True)
    print('PRESERVED',len(completed),'REMAINING',len(remaining),flush=True)
    with concurrent.futures.ProcessPoolExecutor(max_workers=2,mp_context=multiprocessing.get_context('spawn')) as pool:
        for rec in pool.map(score_task,remaining):completed[task_key(rec['kind'],rec['row'])]=rec
    result={'freeze_sha256':freeze,'models':[],'ridge':[],'persistence':[],
        'evaluation_workers':2,'threads_per_worker':1,'resumed_by_model':True}
    for kind,r,_ in tasks:result[kind].append(completed[task_key(kind,r)]['row'])
    b.dump(OUT/'results.json',result)


if __name__=='__main__':run()
