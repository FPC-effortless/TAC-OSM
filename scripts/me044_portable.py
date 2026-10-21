#!/usr/bin/env python3
"""Separate canonical AVX2 serving references. Never rewrite original predictions."""
import argparse,json,lzma,tarfile,io
from pathlib import Path
import numpy as np
import torch
from scripts.me044_train_multiscale import DOMAINS,build_model,predict,period,save,sha

PROFILE={'ATEN_CPU_CAPABILITY':'avx2','MKL_ENABLE_INSTRUCTIONS':'AVX2','ONEDNN_MAX_CPU_ISA':'AVX2'}
torch.set_num_threads(1)

@torch.no_grad()
def record(source,destination):
    import os
    assert all(os.environ.get(k)==v for k,v in PROFILE.items())
    assert torch.backends.cpu.get_cpu_capability()=='AVX2'
    source=Path(source);dest=Path(destination);dest.mkdir(parents=True,exist_ok=False)
    original=json.loads((source/'results.json').read_text());assert original['completed']
    report={'variant':'AVX2_SERVING','original_training_source':original['source_commit'],
        'profile':PROFILE,'rows':[],'new_fits':0,'original_cross_ISA_exact_replay':'FAILED_RETAINED'}
    for domain,data in original['domains'].items():
        mod=DOMAINS[domain];z,_=mod.get_source(source/domain)
        for row in data['fits']:
            cp=source/domain/row['checkpoint'];assert sha(cp)==row['checkpoint_sha256']
            net=build_model(row['arm'],row['horizon'],row['seed'],period(domain))
            net.load_state_dict(torch.load(cp,weights_only=True)['state_dict']);net.eval()
            with np.load(cp.parent/'DEV.npz') as old:
                ps=[];ys=[];qs=[];cs=[];origins=old['origin']
                for j in range(0,len(origins),16):
                    x,y=mod.samples(z,origins[j:j+16],row['horizon']);p,q,c=predict(net,x)
                    ps.append(p.numpy());ys.append(y.numpy())
                    if q is not None:qs.append(q.numpy())
                    if c is not None:cs.append(c.numpy())
                p=np.concatenate(ps);y=np.concatenate(ys)
                q=np.concatenate(qs) if qs else np.empty((0,1),np.float32)
                c=np.concatenate(cs) if cs else np.empty((0,),np.float32)
                assert np.array_equal(y,old['truth'])
                drift=np.abs(p.astype('float64')-old['pred'].astype('float64'))
                e=np.ascontiguousarray(p,dtype=np.float64)-np.ascontiguousarray(y,dtype=np.float64)
                mse=(e*e).mean((0,1));mae=np.abs(e).mean((0,1));start=-3 if domain=='tetouan' else -1
                folder=dest/domain/cp.parent.name;folder.mkdir(parents=True)
                path=folder/'DEV.npz';np.savez_compressed(path,pred=p,truth=y,q=q,cheap=c,origin=origins)
                report['rows'].append({'domain':domain,'arm':row['arm'],'lane':row['lane'],
                    'horizon':row['horizon'],'seed':row['seed'],'checkpoint_sha256':row['checkpoint_sha256'],
                    'reference':str(path.relative_to(dest)),'sha256':sha(path),
                    'metric':{'channel_MSE':mse.tolist(),'channel_MAE':mae.tolist(),'primary_MSE':float(mse[start:].mean())},
                    'original_forecast_different_values':int(np.count_nonzero(drift)),
                    'original_forecast_max_absolute_drift':float(drift.max())})
    assert len(report['rows'])==80
    save(dest/'records.json',report)
    print(json.dumps({'status':'RECORDED_SEPARATE_AVX2_VARIANT','models':80,'new_fits':0,
        'max_drift':max(r['original_forecast_max_absolute_drift'] for r in report['rows']),
        'different_forecast_values':sum(r['original_forecast_different_values'] for r in report['rows'])}),flush=True)


def package(source,dest):
    root=Path(source);dest=Path(dest);dest.mkdir(parents=True,exist_ok=True)
    report=json.loads((root/'records.json').read_text());manifest={'study':'ME044PORTABLE','archives':[],'profile':PROFILE}
    for domain in ('tetouan','etth1'):
        for lane in ('exposure','cpu'):
            paths=[root/'records.json']+[root/r['reference'] for r in report['rows'] if r['domain']==domain and r['lane']==lane]
            entries=[{'path':str(p.relative_to(root)),'sha256':sha(p),'bytes':p.stat().st_size} for p in paths]
            archive=dest/f'me044-avx2-{domain}-{lane}.tar.xz'
            with lzma.open(archive,'wb',filters=[{'id':lzma.FILTER_LZMA2,'dict_size':1<<26,'preset':6}]) as stream:
                with tarfile.open(fileobj=stream,mode='w|') as tf:
                    for p in paths:tf.add(p,arcname=str(p.relative_to(root)),recursive=False)
                    b=json.dumps({'payloads':entries},indent=2).encode();info=tarfile.TarInfo('INDEX.json');info.size=len(b);tf.addfile(info,io.BytesIO(b))
            assert archive.stat().st_size<49_000_000
            manifest['archives'].append({'path':archive.name,'sha256':sha(archive),'bytes':archive.stat().st_size,'payloads':len(entries)})
    save(dest/'me044-manifest.json',manifest);print(json.dumps(manifest),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['record','package']);p.add_argument('source');p.add_argument('destination');a=p.parse_args()
    if a.mode=='record':record(a.source,a.destination)
    else:package(a.source,a.destination)
