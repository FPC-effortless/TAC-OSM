#!/usr/bin/env python3
"""Copy only six byte-verified LANGTRAIN025 original assets; never unpickle."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

ZIP_SHA='87ebc3e69dbf0526d8093b264c37fe0e6c8a65ff04c10c4f66b7bfe8233e31e5'
RESULT_SHA='d25d61a8ae848227cdbf9424854e51a379afdcc3c0dc79c0bbbf160f2e105458'
RESULT='LANGTRAIN025_AUTHENTIC_EXPOSED_DEV_RECEIPT.json'


def sha(raw):return hashlib.sha256(raw).hexdigest()


def preserve(archive,destination,root):
    archive,destination,root=map(Path,(archive,destination,root))
    if sha(archive.read_bytes())!=ZIP_SHA:raise ValueError('first-fit artifact ZIP hash differs')
    expected={RESULT,*[f'byte_waveced_seed_{s}.pt' for s in range(1701,1706)]}
    with zipfile.ZipFile(archive) as z:
        names=z.namelist()
        if len(names)!=len(set(names)) or set(names)!=expected:raise ValueError('original artifact membership differs')
        if any(x.file_size>10*1024*1024 for x in z.infolist()):raise ValueError('unexpected original payload size')
        raw=z.read(RESULT)
        if sha(raw)!=RESULT_SHA:raise ValueError('original numerical receipt differs')
        receipt=json.loads(raw)
        if sorted(r['seed'] for r in receipt['model_runs'])!=list(range(1701,1706)):raise ValueError('original seed coverage differs')
        for name,digest in receipt['code_sha256'].items():
            if name not in ('byte_waveced_500k.py','train_byte_native.py'):raise ValueError('unrecognized original source')
            if sha((root/'research/mathematical-engineering/langtrain025'/name).read_bytes())!=digest:
                raise ValueError('original training/model source changed')
        payloads={RESULT:raw}
        for row in receipt['model_runs']:
            name=row['checkpoint']
            if name!=f"byte_waveced_seed_{row['seed']}.pt":raise ValueError('original checkpoint path differs')
            blob=z.read(name)
            if sha(blob)!=row['checkpoint_sha256']:raise ValueError('original checkpoint hash differs')
            payloads[name]=blob
    destination.mkdir(parents=True,exist_ok=True)
    for name,blob in payloads.items():
        path=destination/name
        if path.is_symlink() or (path.exists() and path.read_bytes()!=blob):
            raise ValueError('refuse to replace original evidence')
    for name,blob in payloads.items():
        path=destination/name
        if not path.exists():path.write_bytes(blob)
    provenance=dict(original_run=40756443660,original_artifact=12846836297,zip_sha256=ZIP_SHA,
        original_result_sha256=RESULT_SHA,files={n:dict(bytes=len(b),sha256=sha(b)) for n,b in payloads.items()},
        evidence='ORIGINAL_BYTES_PRESERVED_NOT_SCORE_REPLAY',no_pickle_loading=True)
    (destination/'ARCHIVAL_PROVENANCE.json').write_text(json.dumps(provenance,indent=2,sort_keys=True)+'\n')
    return provenance


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--archive',type=Path,required=True)
    p.add_argument('--destination',type=Path,required=True);p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]);a=p.parse_args()
    preserve(a.archive,a.destination,a.root)
