#!/usr/bin/env python3
"""Lossless original model delivery with bounded, indexed standard-library XZ."""
import argparse,hashlib,io,json,lzma,tarfile
from pathlib import Path

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,obj):Path(p).write_text(json.dumps(obj,indent=2,sort_keys=True)+'\n')

def package(source,destination):
    root=Path(source);dest=Path(destination);dest.mkdir(parents=True,exist_ok=True)
    report=json.loads((root/'results.json').read_text());assert report['completed']
    manifest={'study':'ME044','original_source_commit':report['source_commit'],
        'format':'tar.xz; LZMA2 dict64MiB; INDEX.json SHA256 of every original payload','archives':[]}
    for domain in report['domains']:
        for lane in ('exposure','cpu'):
            for seed in (44001,44002):
                paths=[root/f for f in ('results.json','audit.json','latency.json','inference.json')]
                paths.extend(sorted(root.glob('*.log')))
                paths.append(root/domain/'source-provenance.json')
                paths.extend(sorted((root/domain).glob(f'parent-*-s{seed}/*')))
                paths.extend(sorted((root/domain).glob(f'{lane}-*-s{seed}-*/*')))
                assert all(p.is_file() for p in paths)
                entries=[{'path':str(p.relative_to(root)),'bytes':p.stat().st_size,'sha256':sha(p)} for p in paths]
                archive=dest/f'me044-{domain}-{lane}-s{seed}.tar.xz'
                with lzma.open(archive,'wb',filters=[{'id':lzma.FILTER_LZMA2,'dict_size':1<<26,'preset':6}]) as stream:
                    with tarfile.open(fileobj=stream,mode='w|') as tf:
                        for p in paths:tf.add(p,arcname=str(p.relative_to(root)),recursive=False)
                        b=json.dumps({'payloads':entries},indent=2,sort_keys=True).encode()
                        info=tarfile.TarInfo('INDEX.json');info.size=len(b);tf.addfile(info,io.BytesIO(b))
                if archive.stat().st_size>=49_000_000:raise ValueError('shard exceeds Git recommended-size budget')
                manifest['archives'].append({'path':archive.name,'bytes':archive.stat().st_size,
                    'sha256':sha(archive),'domain':domain,'lane':lane,'seed':seed,'payloads':len(entries)})
                print(json.dumps(manifest['archives'][-1]),flush=True)
    save(dest/'me044-manifest.json',manifest)


def verify(directory,restore):
    root=Path(restore);root.mkdir(parents=True,exist_ok=False)
    source=Path(directory);manifest=json.loads((source/'me044-manifest.json').read_text());count=0
    assert manifest['study']=='ME044' and len(manifest['archives'])==8
    for archive in manifest['archives']:
        p=source/archive['path'];assert sha(p)==archive['sha256']
        with tarfile.open(p,'r:xz') as tf:
            index=json.load(tf.extractfile('INDEX.json'))['payloads']
            assert len(index)==archive['payloads'] and len(tf.getmembers())==len(index)+1
            for entry in index:
                name=entry['path'];m=tf.getmember(name)
                assert m.isfile() and not Path(name).is_absolute() and '..' not in Path(name).parts
                b=tf.extractfile(m).read()
                assert len(b)==entry['bytes'] and hashlib.sha256(b).hexdigest()==entry['sha256']
                out=root/name;out.parent.mkdir(parents=True,exist_ok=True)
                if out.exists():assert out.read_bytes()==b
                else:out.write_bytes(b)
                count+=1
    print(json.dumps({'status':'PASS','archives':8,'payloads':count,'destination':str(root),'refits':0}),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['package','verify']);p.add_argument('source');p.add_argument('destination');a=p.parse_args()
    if a.mode=='package':package(a.source,a.destination)
    else:verify(a.source,a.destination)
