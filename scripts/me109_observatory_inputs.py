"""Immutable original FIT tensor export and fieldwise producer diagnostics.

No fitting, scoring, gate changes, or edits to original audit receipts.
"""
import argparse
import json
import platform
import numpy as np
import torch
from scripts import me109_residual_training as run

ARRAYS = run.OUT/'OBSERVATORY_FIT_INPUTS.npz'
RECEIPT = run.OUT/'OBSERVATORY_FIT_INPUTS.json'
FIELDS = ('values','observed','ages','clock','owners','hours','future_truth','future_mask','current_truth','reconstruction_mask')


def build(d, batch):
    rng = np.random.default_rng(batch['batch_seed'])
    rng.integers(len(d['gap']),size=16); rng.integers(len(d['general']),size=16)
    return run.tensor_batch(d,np.asarray(batch['origins']),256,rng)


def poison_clean(d, batch):
    return run.tensor_batch(d,np.asarray(batch['origins'])[:1],256,
                            np.random.default_rng(batch['batch_seed']))[:6]


def export():
    if ARRAYS.exists() or RECEIPT.exists():
        raise RuntimeError('original export exists; no overwrite')
    d=run.fit_data(); assert d['phase']=='fit'
    original=json.loads((run.OUT/'POSTHOC_FIT_REGIME_GRADIENTS.json').read_text())
    saved=json.loads((run.OUT/'OBSERVATORY_ANTICHEAT.json').read_text())
    arrays={}; batches=[]
    for batch_id in range(3):
        batch=next(r for r in original['checks'] if r['batch']==batch_id)
        x=build(d,batch)
        digest=run.digest([run.sha_tensor(v) for v in x])
        assert digest==batch['input_sha256'], 'export is not original recorded input'
        for row in original['checks']:
            if row['batch']==batch_id: assert row['input_sha256']==digest
        for name,v in zip(FIELDS,x): arrays[f'batch{batch_id}_{name}']=v.numpy()
        batches.append(dict(batch=batch_id,batch_seed=batch['batch_seed'],origins=batch['origins'],input_sha256=digest))
    batch=next(r for r in original['checks'] if r['batch']==0)
    clean=poison_clean(d,batch)
    from tac_osm.research_anticheat import fingerprint
    fingerprint_clean=fingerprint(clean)
    for row in saved['checks']:
        f=next(f for f in row['anticheat_ledger']['findings'] if f['check']=='future_invariance')
        assert f['evidence']['inputs_sha256']==fingerprint_clean
    for name,v in zip(FIELDS,clean): arrays['poison_clean_'+name]=v.numpy()
    arrays['fit_mean']=d['mean']; arrays['fit_std']=d['std']
    arrays['rawscale']=np.nanstd(d['raw'][...,:6].astype(np.float64),axis=(0,1)).clip(1e-4).astype(np.float32)
    np.savez_compressed(ARRAYS,**arrays)
    receipt=dict(scope='original immutable FIT producer tensors; not new evidence or protocol',
        source_archive_sha256=run.data.SHA,original_posthoc_sha256=run.sha(run.OUT/'POSTHOC_FIT_REGIME_GRADIENTS.json'),
        original_observatory_sha256=run.sha(run.OUT/'OBSERVATORY_ANTICHEAT.json'),
        file_sha256=run.sha(ARRAYS),batches=batches,poison_clean_fingerprint=fingerprint_clean,
        tensors={k:dict(sha256=run.sha_tensor(torch.from_numpy(v)),shape=list(v.shape),dtype=str(v.dtype)) for k,v in arrays.items()},
        environment=dict(python=platform.python_version(),numpy=np.__version__,torch=torch.__version__),
        no_fit=True,no_original_overwrite=True)
    run.write(RECEIPT,receipt)
    print('ME109_ORIGINAL_FIT_INPUT_EXPORT_PASS',ARRAYS.stat().st_size)


def load():
    receipt=json.loads(RECEIPT.read_text())
    assert run.sha(ARRAYS)==receipt['file_sha256']
    arrays=np.load(ARRAYS,allow_pickle=False)
    assert set(arrays.files)==set(receipt['tensors'])
    for name,record in receipt['tensors'].items():
        v=arrays[name]
        assert list(v.shape)==record['shape'] and str(v.dtype)==record['dtype']
        assert run.sha_tensor(torch.from_numpy(v))==record['sha256']
    return arrays,receipt


def compare_field(name,actual,expected):
    a=actual.detach().numpy() if isinstance(actual,torch.Tensor) else actual
    b=expected
    row=dict(field=name,current_sha256=run.sha_tensor(torch.from_numpy(a)),original_sha256=run.sha_tensor(torch.from_numpy(b)),
             current_shape=list(a.shape),original_shape=list(b.shape),current_dtype=str(a.dtype),original_dtype=str(b.dtype))
    row['shape_dtype_equal']=a.shape==b.shape and a.dtype==b.dtype
    row['exact']=row['shape_dtype_equal'] and np.array_equal(a,b)
    if row['shape_dtype_equal']:
        row['max_abs_error']=float(np.max(np.abs(a.astype(np.float64)-b.astype(np.float64)))) if a.size else 0.
        row['contract_pass']=bool(np.allclose(a,b,rtol=1e-5,atol=1e-6)) if np.issubdtype(a.dtype,np.floating) else row['exact']
    else: row['max_abs_error']=None;row['contract_pass']=False
    return row


def diagnose():
    arrays,receipt=load();d=run.fit_data();rows=[]
    for batch in receipt['batches']:
        for name,v in zip(FIELDS,build(d,batch)):
            rows.append(compare_field(f"batch{batch['batch']}_{name}",v,arrays[f"batch{batch['batch']}_{name}"]))
    for name,v in zip(FIELDS,poison_clean(d,receipt['batches'][0])):
        rows.append(compare_field('poison_clean_'+name,v,arrays['poison_clean_'+name]))
    for name,v in (('fit_mean',d['mean']),('fit_std',d['std']),('rawscale',np.nanstd(d['raw'][...,:6].astype(np.float64),axis=(0,1)).clip(1e-4).astype(np.float32))):
        rows.append(compare_field(name,v,arrays[name]))
    result=dict(status='PASS' if all(r['contract_pass'] for r in rows) else 'FAIL',
                source_archive_sha256=run.data.SHA,fields=rows)
    print(json.dumps(result,indent=2))
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('export','diagnose'));args=p.parse_args()
    export() if args.mode=='export' else diagnose()
