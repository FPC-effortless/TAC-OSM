"""Constructed engineering checks; no publisher scoring, fit claims or GPU use."""
import copy
import json
from pathlib import Path
import runpy
import shutil
import sys
import zipfile

import numpy as np
import pytest
import torch

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from protocol004 import (ARMS,CELLS,MILESTONES,ROOT,SEEDS,batches,learning_rate,make,scoring,select_checkpoint)
from evaluate004 import suggestions,summarize
from package004 import package,bootstrap
from run004 import assignments
from scripts.validate_research_parity import ParityError,check_manifest

MANIFEST=json.loads((ROOT/'research/mathematical-engineering/codevalue004/FAIR_COMPARISON.json').read_text())

def test_schedule_endpoints_and_monotonicity():
    assert {learning_rate('small_constant',i) for i in range(5120)}=={.001}
    rates=[learning_rate('broad_cosine',i) for i in range(5120)]
    assert rates[0]==pytest.approx(.001) and rates[-1]==pytest.approx(.0001)
    assert all(a>=b for a,b in zip(rates,rates[1:]))
    assert all(learning_rate('small_cosine',i)==r for i,r in enumerate(rates))
    with pytest.raises(ValueError):learning_rate('unknown',0)
    with pytest.raises(ValueError):learning_rate('small_cosine',5120)

def test_selection_uses_dev_minimum_and_early_tie():
    curve=[{'step':s,'nll':n} for s,n in zip(MILESTONES,[2.,1.8,1.5,1.5,1.7])]
    assert select_checkpoint(curve)==1280
    with pytest.raises(ValueError):select_checkpoint(curve[:-1])
    curve[-1]['nll']=float('nan')
    with pytest.raises(ValueError):select_checkpoint(curve)

def test_fixed_tasks_balanced_without_duplication():
    jobs=assignments();expected={(a,c,s) for a in ARMS for c in CELLS for s in SEEDS}
    assert len(jobs)==2 and all(len(j)==12 for j in jobs)
    assert {tuple(t) for j in jobs for t in j}==expected
    assert all(sum(t[0]=='tac' for t in j)==6 for j in jobs)

def test_sampler_matches_frozen_original_prefix():
    import train as old
    docs={'a':b'abcd'*100,'b':b'1234'*100}
    actual=[(i,r) for i,r,_ in batches(docs,1701,320)]
    assert actual==list(old.batches(docs,1701))
    assert [r for _,r,_ in batches(docs,1701,2)]==[r for _,r,_ in batches(docs,1701,2)]

def test_actual_inherited_models_are_causal_connected_and_same_size():
    torch.set_num_threads(2);torch.manual_seed(940004)
    x=torch.randint(0,256,(2,32));poison=x.clone();poison[:,16:]=(poison[:,16:]+127)%256
    for arm in ARMS:
        model=make(arm,'cpu').eval()
        with torch.no_grad():assert float((model(x)[:,:16]-model(poison)[:,:16]).abs().max())<=1e-6
        model.train();loss=torch.nn.functional.cross_entropy(model(x[:,:-1]).reshape(-1,256),x[:,1:].reshape(-1))
        loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())

class Repeat(torch.nn.Module):
    def forward(self,x):return torch.nn.functional.one_hot(x,256).float()*20

def test_free_generation_does_not_read_gold_and_full_file_masks():
    torch.set_num_threads(2)
    raw=b'\0'*1024;poison=bytearray(raw);poison[129]=1
    clean,_=suggestions(Repeat(),raw,'cpu');changed,_=suggestions(Repeat(),bytes(poison),'cpu')
    assert len(clean)==512 and clean[0]['position']==128
    assert clean[0]['generated_hex']==changed[0]['generated_hex']=='00'*8
    assert clean[0]['exact']['4']==1 and changed[0]['exact']['4']==0
    r=scoring.neural_file(Repeat(),b'\0'*301,'cpu',completion=False)
    assert r['targets']==237 and r['accuracy']==1

@pytest.mark.parametrize('mutation',['study','size','cell','seed','budget','selection','claims'])
def test_scope_gate_is_exact_and_does_not_relax_global_policy(mutation):
    assert check_manifest(MANIFEST)['eligible_for_architecture_superiority'] is False
    m=copy.deepcopy(MANIFEST)
    if mutation=='study':m['study_id']='CODEVALUE005'
    if mutation=='size':m['models'][0]['trainable_parameters']+=1
    if mutation=='cell':m['models'][0]['cell']='unregistered'
    if mutation=='seed':m['models'][0]['seed']=1704
    if mutation=='budget':m['budget']['training_updates']=10240
    if mutation=='selection':m['budget']['checkpoint_candidates']=[320,5120]
    if mutation=='claims':m['no_architecture_superiority_claim']=False
    with pytest.raises(ParityError):check_manifest(m)

def test_final_gate_requires_complete_paired_budget_and_dev_selection():
    m=copy.deepcopy(MANIFEST)
    for r in m['models']:
        r['measured']={'steps':5120,'targets':5242880,'sample_sha256':'a'*64,
            'selected_step':1280,'selected_checkpoint_sha256':'b'*64,
            'cold_replay_max_abs_difference':0.,'future_poison_max_abs_difference':0.,
            'checkpoint_sha256':'c'*64,'trainable_parameters':r['trainable_parameters'],
            'deployed_parameters':r['deployed_parameters']}
    assert check_manifest(m,'final')['status']=='PASS_SINGLE_MODEL'
    m['models'][0]['measured']['sample_sha256']='d'*64
    with pytest.raises(ParityError):check_manifest(m,'final')

@pytest.mark.parametrize('unpacked',[False,True])
def test_actual_bootstrap_both_mount_forms_and_secret_exclusion(tmp_path,monkeypatch,unpacked):
    data=tmp_path/'data';data.mkdir();(data/'SOURCE_RECEIPT.json').write_text('{}')
    stage=tmp_path/'stage';receipt=package(data,stage,'testaccount',require_clean=False)
    archive=stage/'dataset/bundle.zip'
    inputs=tmp_path/'input';inputs.mkdir();working=tmp_path/'working';working.mkdir()
    with zipfile.ZipFile(archive) as z:
        assert not any('kaggle.json' in n or '.kaggle' in n for n in z.namelist())
        if unpacked:z.extractall(inputs/'dataset')
    if not unpacked:shutil.copy(archive,inputs/'bundle.zip')
    calls=[]
    monkeypatch.setattr(runpy,'run_path',lambda p:{'run':lambda *a,**kw:calls.append((p,a,kw))})
    code=bootstrap(archive).replace('/kaggle/input',str(inputs)).replace('/kaggle/working',str(working))
    exec(compile(code,'bootstrap004','exec'),{})
    assert Path(calls[0][0]).name=='run004.py' and calls[0][2]['max_seconds']==7020
    assert calls[0][1][1]==working/'CODEVALUE004'
    meta=json.loads((stage/'kernel/kernel-metadata.json').read_text())
    assert meta['is_private']=='true' and meta['enable_internet']=='false'
    assert meta['machine_shape']=='NvidiaTeslaT4'
    assert len(json.loads((stage/'dataset/dataset-metadata.json').read_text())['title'])<=50

def test_complete_factorial_summary_is_json_serializable(tmp_path):
    scores=tmp_path/'scores';scores.mkdir()
    def write(name,predictor,cell,seed=None):
        for i in range(10):
            nll=2. if cell=='small_constant' else 1.8 if cell=='small_cosine' else 1.7 if cell=='broad_constant' else 1.5
            r={'predictor':predictor,'corpus':cell.split('_')[0],'source':{'file':str(i),'group':f'group{i//2}'},
               'targets':100,'nll_sum':nll*100,'top1':60,'offers':[{'exact':{'4':1}}]}
            if seed is not None:r.update(cell=cell,seed=seed)
            (scores/f'{name}-{i}.json').write_text(json.dumps(r))
    for a in ARMS:
        for c in CELLS:
            for s in SEEDS:write(f'{a}-{c}-{s}',a,c,s)
    for b in ('p3','wb6'):
        for corpus in ('small','broad'):write(f'{b}-{corpus}',b,corpus)
    summarize(tmp_path,repetitions=12)
    r=json.loads((tmp_path/'RESULTS.json').read_text())
    assert len(r['summary'])==28 and r['tac_primary_generalization_gate']=='PASS_INTERNAL_LANGUAGE_SIGNAL'
    assert r['factorial_effects']['tac']['broad_cosine_vs_small_constant_gain']['mean_nll_gain']==pytest.approx(.5)
    assert r['commercial_viability_proved'] is False
