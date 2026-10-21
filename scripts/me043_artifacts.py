#!/usr/bin/env python3
"""Create and independently verify bounded original ME043 model archives."""
import argparse,hashlib,io,json,tarfile
from pathlib import Path
import numpy as np
import torch

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,v):Path(p).write_text(json.dumps(v,indent=2,sort_keys=True)+'\n')

def package(source,destination):
    root=Path(source);dest=Path(destination);dest.mkdir(parents=True,exist_ok=True)
    report=json.loads((root/'results.json').read_text());assert report['completed']
    manifest={'study':report['study'],'original_source_commit':report['source_commit'],
              'format':'gzip tar, INDEX.json records SHA256 of every payload','archives':[]}
    for domain,data in report['domains'].items():
        for lane in ('exposure','cpu'):
            paths=[root/f for f in ('results.json','replay.json','latency.json','metric-audit.json')]
            paths.extend(sorted(root.glob('*.log')))
            paths.extend(sorted((root/domain).glob('parent-*/*')))
            paths.extend(sorted((root/domain).glob(f'{lane}-*/*')))
            paths.append(root/domain/'source-provenance.json')
            entries=[{'path':str(p.relative_to(root)),'bytes':p.stat().st_size,'sha256':sha(p)} for p in paths]
            archive=dest/f'me043-{domain}-{lane}-original.tar.gz'
            with tarfile.open(archive,'w:gz',compresslevel=6) as tf:
                for p in paths:tf.add(p,arcname=str(p.relative_to(root)),recursive=False)
                b=(json.dumps({'payloads':entries},indent=2,sort_keys=True)+'\n').encode()
                info=tarfile.TarInfo('INDEX.json');info.size=len(b);tf.addfile(info,io.BytesIO(b))
            if archive.stat().st_size>=95_000_000:raise ValueError('archive too large for bounded Git delivery')
            manifest['archives'].append({'path':archive.name,'bytes':archive.stat().st_size,
                'sha256':sha(archive),'payloads':len(entries),'domain':domain,'lane':lane})
    save(dest/'me043-manifest.json',manifest);print(json.dumps(manifest),flush=True)


def verify(directory,restore):
    directory=Path(directory);root=Path(restore);root.mkdir(parents=True,exist_ok=False)
    manifest=json.loads((directory/'me043-manifest.json').read_text());count=0
    for item in manifest['archives']:
        path=directory/item['path'];assert sha(path)==item['sha256']
        with tarfile.open(path,'r:gz') as tf:
            index=json.load(tf.extractfile('INDEX.json'))['payloads']
            assert len(index)==item['payloads']
            assert len(tf.getmembers())==len(index)+1
            for entry in index:
                name=entry['path'];member=tf.getmember(name)
                assert not Path(name).is_absolute() and '..' not in Path(name).parts
                assert member.isfile()
                b=tf.extractfile(member).read()
                assert len(b)==entry['bytes'] and hashlib.sha256(b).hexdigest()==entry['sha256']
                p=root/name;p.parent.mkdir(parents=True,exist_ok=True)
                if p.exists():assert p.read_bytes()==b
                else:p.write_bytes(b)
                count+=1
    print(json.dumps({'status':'ORIGINAL_ARCHIVE_RESTORATION_PASS','payloads_checked':count,
        'archives':len(manifest['archives']),'destination':str(root)}),flush=True)


def metrics(source):
    root=Path(source);report=json.loads((root/'results.json').read_text());assert report['completed']
    count=0;scalars=0;optimizer=0;max_error=0.;max_bound=0.
    for domain,data in report['domains'].items():
        for row in data['fits']:
            folder=(root/domain/row['checkpoint']).parent
            assert sha(folder/'final.pt')==row['checkpoint_sha256']
            ck=torch.load(folder/'final.pt',weights_only=True)
            assert ck['updates']==row['updates']
            assert len(ck['optimizer']['state'])==len(ck['optimizer']['param_groups'][0]['params'])+len(ck['optimizer']['param_groups'][1]['params'])
            steps={int(v['step']) for v in ck['optimizer']['state'].values()}
            if row['arm'].startswith('scheduled_'):
                from tac_osm.me043_scheduled_compute import build_model
                net=build_model(row['arm'],row['horizon'],row['seed'])
                names=[n for n,p in net.named_parameters() if n.startswith('ar.')]
                names += [n for n,p in net.named_parameters() if not n.startswith('ar.')]
                ids=[i for g in ck['optimizer']['param_groups'] for i in g['params']]
                for name,i in zip(names,ids):
                    expected=row['updates']//8 if name.startswith(('memory.','router.')) or name=='native_gain' else row['updates']
                    assert int(ck['optimizer']['state'][i]['step'])==expected,(name,expected)
            else:
                assert steps=={row['updates']}
            assert sha(folder/'inference.pt')==row['inference_sha256']
            with np.load(folder/'DEV.npz') as saved:
                p=saved['pred'];y=saved['truth'];delta=p.astype('float64')-y.astype('float64')
                expected=np.mean(delta**2,axis=(0,1));expected_mae=np.mean(np.abs(delta),axis=(0,1))
                err=float(np.max(np.abs(expected-row['metric']['channel_MSE'])))
                max_error=max(max_error,err)
                # Original forecast tensors and reloaded NPZ arrays can have
                # different strides. Float64 reductions then associate terms
                # differently; bound both paths using operation count, not the
                # observed error. This changes no original forecast tolerance.
                n=p.shape[0]*p.shape[1]
                u=np.finfo(np.float64).eps/2
                gamma=(2*n+8)*u/(1-(2*n+8)*u)
                bound=2*gamma*np.maximum(expected,np.finfo(np.float64).tiny)
                mae_bound=2*gamma*np.maximum(expected_mae,np.finfo(np.float64).tiny)
                max_bound=max(max_bound,float(bound.max()))
                assert np.all(np.abs(expected-row['metric']['channel_MSE'])<=bound)
                assert np.all(np.abs(expected_mae-row['metric']['channel_MAE'])<=mae_bound)
                start=-3 if domain=='tetouan' else -1
                assert abs(float(expected[start:].mean())-row['metric']['primary_MSE'])<=float(bound[start:].max())
                assert int(p.size)==row['metric']['forecast_scalars']
                q=saved['q']
                if q.size:
                    assert ((q>=0)&(q<=1)).all()
                    assert float((q>.5).mean())==row['metric']['native_fraction']
                scalars+=p.size
            optimizer+=1;count+=1
    result={'status':'PASS','original_models':count,'optimizer_states_verified':optimizer,
            'forecast_scalars_recomputed':scalars,'max_metric_absolute_error':max_error,
            'new_fits':0,'new_test_rows':0,'metric_comparison':'float64 operation-count gamma bound',
            'max_derived_MSE_absolute_bound':max_bound,'strict_float64_equality_initial_check':'FAILED_RECORDED'}
    save(root/'metric-audit.json',result);print(json.dumps(result),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();sub=ap.add_subparsers(dest='mode',required=True)
    for name in ('package','verify','metrics'):
        p=sub.add_parser(name);p.add_argument('source')
        if name!='metrics':p.add_argument('destination')
    a=ap.parse_args()
    if a.mode=='package':package(a.source,a.destination)
    elif a.mode=='verify':verify(a.source,a.destination)
    else:metrics(a.source)
