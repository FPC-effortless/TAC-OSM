"""One final scoring pass, independent datasets on two single-thread CPU workers."""
import concurrent.futures, json, multiprocessing
from pathlib import Path
import numpy as np
import torch
import scripts.me036_research as base
from scripts.me036b_continue import OUT, Ridge
from tac_osm.me036_patch_forecast import build_model


class Persistence(torch.nn.Module):
    def __init__(self,h):super().__init__();self.h=h
    def forward(self,x):return x[:,-1:].expand(-1,self.h,-1)


def evaluate_dataset(name):
    torch.set_num_threads(1)
    frozen=json.loads((OUT/'frozen.json').read_text())
    dest=OUT/f'test-{name}.json'
    if dest.exists():
        old=json.loads(dest.read_text())
        assert old['freeze_sha256']==base.sha(OUT/'frozen.json')
        return old
    z,meta=base.load_data(name);_,val,end=meta['split_edges']
    result={'dataset':name,'freeze_sha256':base.sha(OUT/'frozen.json'),'models':[],'ridge':[],'persistence':[]}
    for r in frozen['fits']:
        if r['dataset']!=name or r['status']!='PASS':continue
        model=build_model(r['arm'],r['horizon'],r['seed'])
        model.load_state_dict(torch.load(base.ROOT/r['selected']['checkpoint'],weights_only=False)['state_dict'])
        metric=base.score(model,z,val,end,r['horizon'])
        row={k:r[k] for k in ('key','dataset','horizon','arm','seed','parameters','charged_training_CPU_seconds')}
        row.update({'selected_step':r['selected']['step'],'test':metric})
        result['models'].append(row);base.dump(OUT/f'test-progress-{name}.json',result)
        print('TEST',r['key'],round(metric['MSE'],6),round(metric['CPU_seconds'],2),flush=True)
    for r in frozen['ridge']:
        if r['dataset']!=name:continue
        model=Ridge(np.load(base.ROOT/r['coefficient']))
        metric=base.score(model,z,val,end,r['horizon'])
        result['ridge'].append({**r,'test':metric})
        print('TEST RIDGE',name,r['horizon'],round(metric['MSE'],6),flush=True)
    for h in base.HORIZONS:
        result['persistence'].append({'dataset':name,'horizon':h,'test':base.score(Persistence(h),z,val,end,h)})
    base.dump(dest,result)
    return result


def run():
    torch.set_num_threads(1)
    frozen=json.loads((OUT/'frozen.json').read_text())
    code_freeze=json.loads((base.REPORT/'code-freeze.json').read_text())
    for item in code_freeze['files']:
        assert base.sha(base.ROOT/item['path'])==item['sha256'],item['path']
    assert frozen['protocol_sha256']==base.sha(base.REPORT/'PROTOCOL.md')
    assert frozen['source_manifest_sha256']==base.sha(base.REPORT/'sources.json')
    assert frozen['continuation_protocol_sha256']==base.sha(base.REPORT/'CONTINUATION.md')
    assert frozen['parent_freeze_sha256']==base.sha(base.OUT/'frozen.json')
    # Global verification before any worker opens final test scoring.
    for r in frozen['fits']:
        for e in r['endpoints']:assert base.sha(base.ROOT/e['checkpoint'])==e['sha256']
    for r in frozen['ridge']:assert base.sha(base.ROOT/r['coefficient'])==r['sha256']
    result={'freeze_sha256':base.sha(OUT/'frozen.json'),'models':[],'ridge':[],'persistence':[],
        'evaluation_workers':2,'threads_per_worker':1}
    with concurrent.futures.ProcessPoolExecutor(max_workers=2,mp_context=multiprocessing.get_context('spawn')) as pool:
        for data in pool.map(evaluate_dataset,('electricity','ETTm1','weather','ETTh1','ETTh2')):
            for key in ('models','ridge','persistence'):result[key].extend(data[key])
    base.dump(OUT/'results.json',result)


if __name__=='__main__':run()
